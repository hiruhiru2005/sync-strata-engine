// prefill_mmq_bih3_test - QuantLab (11/12): the prompt path's MMQ products of the bih3 types (180 + K, the points
// computed from the state: x_i = s * C_i, weight j = (T0.bj + T1.bj + T2.bj - 384) / 128, see iq_kernels.cu tq_bih3p)
// with mmq::set_bih3(true): each type runs as its K's table type and the trellis tiles compute the points, rounded to
// 1/32 (third_party/ggml-tq/tq.cuh).  Random codes and fp16 scales in the table types' block (fp16 d + 32 steps of K
// bits: what the prefill's gather_r1q2 writes), gate/up 1280 x 2560 and down 2560 x 640, several experts per product,
// permuted rows, an all-zero row, against a double-precision product of the exact points.  MMQ rounds the activations
// to q8_1 and the tiles the points to 1/32, so this is a screen (rel L2 < 4%), as prefill_mmq_kquant_test.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml-common-tq.h"   // --lut: the built-in fp16 codebook (host data here)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;
int g_bad = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

// one block's 128 exact weights (without d): the states of the table types (K 8: TQ2_T's bytes, else the 15 bits from
// stream bit (31 - t) K, circular), the bih3 points
void bih3_block(const uint8_t* qs, int K, float* y) {
    for (int t = 0; t < 32; ++t) {
        uint32_t s = 0;
        if (K == 8) s = ((uint32_t) qs[(t + 31) % 32] << 8) | qs[t];
        else
            for (int j = 0; j < 15; ++j) {
                const int bit = ((31 - t) * K + j) % (32 * K);
                s |= (uint32_t) ((qs[bit >> 3] >> (bit & 7)) & 1) << j;
            }
        s &= 0x7FFFu;
        const uint32_t x0 = s * 0x9E3779B1u, x1 = s * 0x85EBCA6Bu, x2 = s * 0xC2B2AE35u, x3 = s * 0x27D4EB2Fu;
        const uint32_t T0 = (x0 >> 8) | ((x1 >> 8) & 255u) << 24, T1 = (x1 >> 16) | ((x2 >> 8) & 0xFFFFu) << 16;
        const uint32_t T2 = (x2 >> 24) | (x3 & 0xFFFFFF00u);
        for (int j = 0; j < 4; ++j)
            y[4 * t + j] = (float) ((int) ((T0 >> (8 * j)) & 255u) + (int) ((T1 >> (8 * j)) & 255u) +
                                    (int) ((T2 >> (8 * j)) & 255u) - 384) / 128.0f;
    }
}

std::vector<float> make_x(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * (size_t) cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * (size_t) cols + (size_t) k] =
                0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
    std::fill(x.end() - cols, x.end(), 0.0f);   // an all-zero row: exactly zero out
    return x;
}

std::vector<int32_t> perm(int n, int shift, bool reverse) {
    std::vector<int32_t> p((size_t) n);
    for (int i = 0; i < n; ++i) p[(size_t) i] = (int32_t) (((reverse ? n - 1 - i : i) + shift) % n);
    return p;
}

// --lut: the table types themselves (no set_bih3), against the built-in fp16 codebook - the screen's own yardstick
bool g_lut = false;
int lut_type(int K) { return K == 8 ? 144 : K == 6 ? 145 : K == 7 ? 146 : K == 5 ? 150 : K == 4 ? 151 : 160 + K; }
void lut_block(const uint8_t* qs, int K, float* y) {
    for (int t = 0; t < 32; ++t) {
        uint32_t s = 0;
        if (K == 8) s = ((uint32_t) qs[(t + 31) % 32] << 8) | qs[t];
        else
            for (int j = 0; j < 15; ++j) {
                const int bit = ((31 - t) * K + j) % (32 * K);
                s |= (uint32_t) ((qs[bit >> 3] >> (bit & 7)) & 1) << j;
            }
        const uint32_t x = (s & 0x7FFFu) * 0x9e3779b1u, a = x >> 21, c = (x >> 10) & 2047u;
        const uint16_t* L = (const uint16_t*) tq_lut_f16;
        const uint16_t v[4] = {L[2 * a], L[2 * a + 1], L[2 * c], L[2 * c + 1]};
        for (int j = 0; j < 4; ++j) {
            __half h;
            std::memcpy(&h, &v[j], 2);
            y[4 * t + j] = __half2float(h);
        }
    }
}

void product(mmq::Context& ctx, cudaStream_t s, int K, const char* role, int64_t out_rows, int64_t cols,
             const std::vector<int>& counts, int trial) {
    const int t = g_lut ? lut_type(K) : 180 + K;
    const std::string name = std::string(g_lut ? "table" : "bih3") + " K" + std::to_string(K) + " " + role + "-" + std::to_string(trial);
    if (!mmq::supported(t) || !mmq::fits(t, out_rows)) throw std::runtime_error(name + ": not covered by MMQ");
    const int n = (int) counts.size(), rows = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int32_t> bounds((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) bounds[(size_t) e + 1] = bounds[(size_t) e] + counts[(size_t) e];
    const auto src = perm(rows, trial + 1, false), dst = perm(rows, trial + 2, true);
    const std::vector<float> x = make_x(rows, cols, trial);
    const size_t bb = 2 + 4 * (size_t) K, nb = (size_t) (cols / 128), eb = mmq::matrix_bytes(t, out_rows, cols);
    if (eb != (size_t) out_rows * nb * bb) throw std::runtime_error(name + ": matrix_bytes");
    std::mt19937 rng(1112u + (uint32_t) K * 31u + (uint32_t) trial);
    std::uniform_real_distribution<float> sd(0.004f, 0.02f);
    std::vector<uint8_t> w((size_t) n * eb + 4096, 0);
    for (int e = 0; e < n; ++e)
        for (size_t b = 0; b < (size_t) out_rows * nb; ++b) {
            uint8_t* blk = w.data() + (size_t) e * eb + b * bb;
            const __half h = __float2half(rng() & 1 ? sd(rng) : -sd(rng));
            std::memcpy(blk, &h, 2);
            for (size_t j = 2; j < bb; ++j) blk[j] = (uint8_t) rng();
        }
    Dev dx(x.size() * 4), dsrc((size_t) rows * 4), ddst((size_t) rows * 4), db(bounds.size() * 4), dw(w.size()),
        dxq(mmq::q8_bytes(rows, cols)), dy((size_t) rows * (size_t) out_rows * 4);
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(dsrc.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(ddst.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemset(dy.p, 0xff, (size_t) rows * (size_t) out_rows * 4), "sentinel");
    ck(cudaDeviceSynchronize(), "inputs");
    mmq::quantize((const float*) dx.p, (const int32_t*) dsrc.p, dxq.p, t, cols, cols, rows, s);
    mmq::Product p;
    p.w = dw.p; p.type = t; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = eb; p.n = n;
    p.xq = dxq.p; p.bounds = (const int32_t*) db.p; p.ids = (const int32_t*) ddst.p; p.total_rows = rows;
    p.max_rows = *std::max_element(counts.begin(), counts.end()); p.dst = (float*) dy.p; p.ld_dst = out_rows;
    ctx.run(p, s);
    ck(cudaGetLastError(), "launch");
    ck(cudaStreamSynchronize(s), "sync");
    std::vector<float> got((size_t) rows * (size_t) out_rows);
    ck(cudaMemcpy(got.data(), dy.p, got.size() * 4, cudaMemcpyDeviceToHost), "y");

    std::vector<float> ref(got.size(), 0.0f), wd((size_t) out_rows * (size_t) cols);
    for (int e = 0; e < n; ++e) {
        for (int64_t o = 0; o < out_rows; ++o)
            for (size_t b = 0; b < nb; ++b) {
                const uint8_t* blk = w.data() + (size_t) e * eb + ((size_t) o * nb + b) * bb;
                __half h;
                std::memcpy(&h, blk, 2);
                const float d = __half2float(h);
                float* y = wd.data() + (size_t) o * (size_t) cols + b * 128;
                if (g_lut) lut_block(blk + 2, K, y);
                else bih3_block(blk + 2, K, y);
                for (int i = 0; i < 128; ++i) y[i] *= d;
            }
        for (int r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
            const float* xr = x.data() + (size_t) src[(size_t) r] * (size_t) cols;
            for (int64_t o = 0; o < out_rows; ++o) {
                double acc = 0;
                const float* wr = wd.data() + (size_t) o * (size_t) cols;
                for (int64_t k = 0; k < cols; ++k) acc += (double) wr[k] * xr[k];
                ref[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o] = (float) acc;
            }
        }
    }
    double err2 = 0, ref2 = 0, max_abs = 0, zero_max = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error(name + ": unwritten or non-finite output");
        const double dd = (double) got[i] - ref[i];
        err2 += dd * dd; ref2 += (double) ref[i] * ref[i]; max_abs = std::max(max_abs, std::fabs(dd));
    }
    for (int r = 0; r < rows; ++r)
        if (src[(size_t) r] == rows - 1)
            for (int64_t o = 0; o < out_rows; ++o)
                zero_max = std::max(zero_max, (double) std::fabs(got[(size_t) dst[(size_t) r] * (size_t) out_rows + (size_t) o]));
    const double rms = std::sqrt(ref2 / (double) ref.size()), rel = std::sqrt(err2 / (double) ref.size()) / rms;
    std::printf("%-24s rows %2d experts %d  ref_rms %.4g  rel_l2 %.5f  max/rms %.4f  zero row %.3g\n", name.c_str(), rows, n,
                rms, rel, max_abs / rms, zero_max);
    if (rel > 0.04 || max_abs / rms > 0.35 || zero_max > std::max(1e-5, 1e-4 * rms)) {
        std::printf("  ^ outside the screen\n");
        ++g_bad;
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        g_lut = argc > 1 && std::string(argv[1]) == "--lut";
        if (!mmq::built()) { std::printf("no MMQ in this build\n"); return 77; }
        if (mmq::supported(188)) throw std::runtime_error("a bih3 type is an MMQ type before set_bih3");
        std::string err;
        if (!g_lut && !mmq::set_bih3(true, err)) { std::printf("set_bih3: %s\n", err.c_str()); return 77; }
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        {
            mmq::Context ctx;
            const std::vector<std::vector<int>> batches{{1, 3, 3}, {17}};
            int trial = 0;
            for (const auto& counts : batches) {
                for (int K : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) {
                    product(ctx, s, K, "gate/up", 1280, 2560, counts, trial);
                    product(ctx, s, K, "down", 2560, 640, counts, trial + 7);
                }
                ++trial;
            }
        }
        ck(cudaStreamDestroy(s), "destroy");
        if (g_bad) {
            std::printf("prefill MMQ bih3 screen: %d products outside\n", g_bad);
            return 1;
        }
        std::printf("prefill MMQ bih3 screen passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill MMQ bih3 screen failed: %s\n", e.what());
        return 1;
    }
}
