// ql_fuse_parity_test - QuantLab (11/13): the decode's folded launches against the launches they replace, byte for byte.
//   STRATA_QL_FUSE_HADQ: hadamard_q8_1_rows against hadamard_rows + quantize_q8_1_rows (the rotated floats and the q8_1
//     blocks), at the verify window's expert input (T x n_embd rows) and the expert path's hidden rows (T*k x n_ff, the
//     tq-mix path's cap_entries), T 1..8, Hadamard blocks 32..256, in place and out of place.
//   STRATA_QL_FUSE_HITS: native_moe_combine_multi_hits_added against moe_hit_add + native_moe_combine_multi (k = 10),
//     T 2..8 (the verify path folds only n > 1), random hit subsets (none, some, all), parts random or zero.
// Exit 77 without a CUDA device.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace K = strata::kernels;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n ? n : 16), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <class T> T* as() const { return (T*) p; }
};

template <class T> std::vector<T> down(const Dev& d, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

// returns the number of failures
int hadq_case(int64_t rows, int64_t width, int block, bool in_place, bool with_signs, uint32_t seed, cudaStream_t s) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(1e-3f, 50.f);
    std::vector<float> x((size_t) (rows * width)), sg((size_t) width);
    for (int64_t r = 0; r < rows; ++r) {
        const float m = mag(rng);
        for (int64_t c = 0; c < width; ++c) x[(size_t) (r * width + c)] = nd(rng) * m;
    }
    if (rows > 2) {   // a zero row and a row with one huge value
        std::fill(x.begin() + width, x.begin() + 2 * width, 0.0f);
        x[(size_t) (2 * width + 7)] = 3e4f;
    }
    for (auto& v : sg) v = (rng() & 1) ? 1.0f : -1.0f;
    const size_t fb = x.size() * 4, qb = (size_t) (rows * (width / 32)) * 36;
    Dev dx0(fb), dx1(fb), dy0(fb), dy1(fb), dq0(qb), dq1(qb), ds(sg.size() * 4);
    ck(cudaMemcpy(dx0.p, x.data(), fb, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dx1.p, x.data(), fb, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(ds.p, sg.data(), sg.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemset(dy0.p, 0x5a, fb), "memset");
    ck(cudaMemset(dy1.p, 0x5a, fb), "memset");
    ck(cudaMemset(dq0.p, 0xa5, qb), "memset");
    ck(cudaMemset(dq1.p, 0xa5, qb), "memset");
    const float* signs = with_signs ? ds.as<float>() : nullptr;
    float* y0 = in_place ? dx0.as<float>() : dy0.as<float>();
    float* y1 = in_place ? dx1.as<float>() : dy1.as<float>();
    // the two launches, as verify.cpp and native_expert_grouped make them
    K::hadamard_rows(dx0.as<float>(), y0, rows, width, block, signs, s);
    K::quantize_q8_1_rows(y0, rows, width, dq0.p, s);
    K::hadamard_q8_1_rows(dx1.as<float>(), y1, dq1.p, rows, width, block, signs, s);
    ck(cudaStreamSynchronize(s), "run");
    const auto a = down<uint8_t>(in_place ? dx0 : dy0, fb), b = down<uint8_t>(in_place ? dx1 : dy1, fb);
    const auto qa = down<uint8_t>(dq0, qb), qbv = down<uint8_t>(dq1, qb);
    const bool ok_y = std::memcmp(a.data(), b.data(), fb) == 0, ok_q = std::memcmp(qa.data(), qbv.data(), qb) == 0;
    if (!ok_y || !ok_q) {
        std::printf("FAIL hadq rows %lld width %lld block %d %s%s: rotated %s, q8_1 %s\n", (long long) rows,
                    (long long) width, block, in_place ? "in place" : "out of place", with_signs ? "" : " no signs",
                    ok_y ? "same" : "DIFFERENT", ok_q ? "same" : "DIFFERENT");
        return 1;
    }
    return 0;
}

int hits_case(int T, int64_t N, int mode, bool zero_parts, uint32_t seed, cudaStream_t s) {
    constexpr int k = 10;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t ne = (size_t) T * k * (size_t) N;
    std::vector<float> parts(ne), hits(ne), w((size_t) T * k), sh((size_t) T * N);
    for (auto& v : parts) v = zero_parts ? 0.0f : nd(rng);
    for (auto& v : hits) v = nd(rng) * 3.0f;
    for (auto& v : w) v = std::uniform_real_distribution<float>(0.0f, 0.3f)(rng);
    for (auto& v : sh) v = nd(rng);
    std::vector<int32_t> all((size_t) T * k);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    const int nh = mode == 0 ? 0 : mode == 2 ? T * k : (int) (rng() % (uint32_t) (T * k));
    std::vector<int32_t> dst(all.begin(), all.begin() + nh);
    dst.resize((size_t) T * k, -1);   // the tail past the count is never read
    const int32_t cnt = nh;
    Dev dp0(ne * 4), dp1(ne * 4), dh(ne * 4), dw(w.size() * 4), dsh(sh.size() * 4), dd(dst.size() * 4), dc(4),
        do0((size_t) T * N * 4), do1((size_t) T * N * 4);
    ck(cudaMemcpy(dp0.p, parts.data(), ne * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dp1.p, parts.data(), ne * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dh.p, hits.data(), ne * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dw.p, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dsh.p, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dd.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dc.p, &cnt, 4, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemset(do0.p, 0x5a, (size_t) T * N * 4), "memset");
    ck(cudaMemset(do1.p, 0x5a, (size_t) T * N * 4), "memset");
    if (!K::native_moe_combine_hits_supported(dp1.as<float>(), dsh.as<float>(), do1.as<float>(), N, k, T)) {
        std::printf("FAIL hits T %d N %lld: the folded combine says it does not cover the case\n", T, (long long) N);
        return 1;
    }
    // the two launches, as verify.cpp makes them (cap = the window's T * k entries)
    K::moe_hit_add(dp0.as<float>(), dh.as<float>(), dd.as<int32_t>(), dc.as<int32_t>(), (int64_t) T * k, N, s);
    K::native_moe_combine_multi(dp0.as<float>(), dw.as<float>(), dsh.as<float>(), do0.as<float>(), N, k, T, s);
    K::native_moe_combine_multi_hits_added(dp1.as<float>(), dh.as<float>(), dd.as<int32_t>(), dc.as<int32_t>(),
                                           dw.as<float>(), dsh.as<float>(), do1.as<float>(), N, k, T, s);
    ck(cudaStreamSynchronize(s), "run");
    const auto a = down<uint8_t>(do0, (size_t) T * N * 4), b = down<uint8_t>(do1, (size_t) T * N * 4);
    if (std::memcmp(a.data(), b.data(), a.size()) != 0) {
        std::printf("FAIL hits T %d N %lld hits %d%s: the outputs differ\n", T, (long long) N, nh, zero_parts ? " zero parts" : "");
        return 1;
    }
    return 0;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        std::printf("ql_fuse_parity_test: no CUDA device, skipped\n");
        return 77;
    }
    try {
        cudaStream_t s = nullptr;
        ck(cudaStreamCreate(&s), "stream");
        int fails = 0, cases = 0;
        uint32_t seed = 1113;
        // the expert input (T rows of n_embd) and the hidden rows (T * 10 entries of n_ff), for both model widths
        const int64_t widths[] = {2048, 2560, 512, 640};
        const int blocks[] = {32, 64, 128, 256};
        for (int T = 1; T <= 8; ++T)
            for (const int64_t wdt : widths)
                for (const int blk : blocks) {
                    if (wdt % blk != 0) continue;
                    const int64_t rows = wdt >= 2048 ? T : (int64_t) T * 10;
                    for (int v = 0; v < 3; ++v) {
                        fails += hadq_case(rows, wdt, blk, v == 1, v != 2, seed++, s);
                        ++cases;
                    }
                }
        // a cap larger than the tokens' entries (native_expert_grouped quantizes all cap_entries rows)
        fails += hadq_case(8 * 10 * 4, 640, 128, true, true, seed++, s); ++cases;
        std::printf("hadq: %d cases, %d failed\n", cases, fails);
        int hfails = 0, hcases = 0;
        for (int T = 2; T <= 8; ++T)
            for (const int64_t N : {2048, 2560})
                for (int mode = 0; mode < 3; ++mode)
                    for (int z = 0; z < 2; ++z) {
                        hfails += hits_case(T, N, mode, z == 1, seed++, s);
                        ++hcases;
                    }
        std::printf("hits: %d cases, %d failed\n", hcases, hfails);
        cudaStreamDestroy(s);
        fails += hfails;
        std::printf("%s\n", fails == 0 ? "ql_fuse_parity_test: PASS" : "ql_fuse_parity_test: FAIL");
        return fails == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("ql_fuse_parity_test: error: %s\n", e.what());
        return 1;
    }
}
