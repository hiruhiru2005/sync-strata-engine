// mmvq_t_bench - QuantLab (11/13) STRATA_QL_MMVQ_T: the decode's token-mixing projections (GDN qkv / attn_gate, the
// attention q / k / v, the output projections) at the engine's shapes, today's kernels (native_mmvq_il with the card's
// table, which falls back to native_mmvq: the single-column kernel at T = 1, b6_mmvq_rows with STRATA_B6_MMVQ_ROWS=1)
// against the native_mmvq_il kernel variants (pf 0 / 2-7, 7 = STRATA_QL_MMVQ_T, rows a warp 1 / 2 / 4): every output compared byte for byte
// with today's for T tokens, then each one's time per call with the weights rotating over enough copies to be cold
// (a layer's weights are cold in the engine; the copies add up to at least --cold-mb): us a call (events, the mean of
// --iters, the best of 3 rounds) and GB/s of the matrix's bytes, next to the read-only yardstick (ql_bw_probe) on the
// same bytes.
//
//     build/mmvq_t_bench [--t A B] [--iters n] [--cold-mb m] [--shape name] [--pf list] [--rows list] [--seed s]
// --shape: qkv | gate | out | q | kv | router (default every one); --pf / --rows: comma lists (default 0,7 and 1,2,4; pf 7 = STRATA_QL_MMVQ_T's kernel,
// pf -1 rows 0 = the environment's choice, e.g. STRATA_QL_MMVQ_T=1 as the engine takes it);
// --b6: native_mmvq's B6 modes to time too (default 0,1,4,8; STRATA_NO_MMVQ_ROWS2=1 turns mode 0 into one row a block).
// --pair: also two matrices of the shape in one launch (native_mmvq_pair_kq) against two launches.
// Exit 1 when any variant differs from today's output.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace strata::kernels {   // QuantLab (11/12) iq_kernels.cu: the bandwidth yardstick (tq_mix_parity --bw)
void ql_bw_probe(const unsigned long long* d_ptrs, const unsigned long long* d_sizes, int n, int blocks_x, int unroll,
                 unsigned* d_out, cudaStream_t s);
}

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("mmvq_t_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(3);
    }
}
struct Shape { const char* name; int type, n_in, n_out; };
std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string it;
    while (std::getline(ss, it, ',')) v.push_back(std::atoi(it.c_str()));
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int t_lo = 1, t_hi = 4, iters = 200, cold_mb = 96, seed = 7;
    std::string only;
    bool pair = false;
    std::vector<int> pfs = {0, 7}, rowsv = {1, 2, 4}, b6v = {0, 1, 4, 8};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--t" && i + 2 < argc) { t_lo = std::atoi(argv[++i]); t_hi = std::atoi(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc) iters = std::max(10, std::atoi(argv[++i]));
        else if (a == "--cold-mb" && i + 1 < argc) cold_mb = std::max(0, std::atoi(argv[++i]));
        else if (a == "--shape" && i + 1 < argc) only = argv[++i];
        else if (a == "--pf" && i + 1 < argc) pfs = parse_list(argv[++i]);
        else if (a == "--rows" && i + 1 < argc) rowsv = parse_list(argv[++i]);
        else if (a == "--b6" && i + 1 < argc) b6v = parse_list(argv[++i]);
        else if (a == "--pair") pair = true;
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { std::printf("mmvq_t_bench: unknown argument %s\n", a.c_str()); return 3; }
    }
    t_lo = std::max(1, std::min(8, t_lo));
    t_hi = std::max(t_lo, std::min(8, t_hi));
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("mmvq_t_bench: no CUDA device, skipped\n");
        return 77;
    }
    K::native_mmvq_set_b6_rows(1);   // the engine's sm_86 default (nv_defaults.cpp)
    const Shape shapes[] = {
        {"qkv", 13, 2560, 10240}, {"gate", 12, 2560, 6144}, {"out", 13, 6144, 2560}, {"q", 13, 2560, 12288},
        {"kv", 13, 2560, 512},
    };
    std::mt19937 rng((unsigned) seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    unsigned* d_out;
    ck(cudaMalloc((void**) &d_out, 64), "malloc");
    int bad = 0, cases = 0;
    for (const Shape& sh : shapes) {
        if (!only.empty() && only != sh.name) continue;
        const size_t bytes = K::native_mmvq_weight_bytes(sh.type, sh.n_in, sh.n_out);
        const int bsz = sh.type == 13 ? 176 : 144;
        const int ncopy = std::max(1, (int) ((size_t) cold_mb * 1000000 / bytes) + 1);
        std::vector<uint8_t> hw(bytes);
        std::vector<uint8_t*> dw(ncopy);
        for (int c = 0; c < ncopy; ++c) {
            for (auto& b : hw) b = (uint8_t) rng();
            for (size_t b = 0; b < bytes / bsz; ++b) {   // d, dmin: finite fp16 in [1e-3, 1e-2)
                const __half d = __float2half(0.001f + 0.009f * (float) (rng() % 1000) / 1000.f);
                const __half m = __float2half(0.001f + 0.009f * (float) (rng() % 1000) / 1000.f);
                std::memcpy(&hw[b * bsz], &d, 2);
                std::memcpy(&hw[b * bsz + 2], &m, 2);
            }
            ck(cudaMalloc((void**) &dw[c], bytes), "malloc w");
            cudaMemcpy(dw[c], hw.data(), bytes, cudaMemcpyHostToDevice);
        }
        // the yardstick: one read-only launch a call, the copies rotating (best blocks_x of a few)
        float probe_us = 1e30f;
        int probe_bx = 0;
        {
            std::vector<unsigned long long> hp(ncopy), hz(ncopy, bytes);
            for (int c = 0; c < ncopy; ++c) hp[c] = (unsigned long long) dw[c];
            unsigned long long *dpp, *dzz;
            ck(cudaMalloc((void**) &dpp, 8 * ncopy), "malloc");
            ck(cudaMalloc((void**) &dzz, 8 * ncopy), "malloc");
            cudaMemcpy(dpp, hp.data(), 8 * ncopy, cudaMemcpyHostToDevice);
            cudaMemcpy(dzz, hz.data(), 8 * ncopy, cudaMemcpyHostToDevice);
            for (int bx : {68, 136, 204, 272, 408, 544}) {
                float best = 1e30f;
                for (int round = 0; round < 3; ++round) {
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < iters; ++i) K::ql_bw_probe(dpp + i % ncopy, dzz + i % ncopy, 1, bx, 4, d_out, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms = 0.f;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best = std::min(best, ms * 1000.f / iters);
                }
                if (best < probe_us) { probe_us = best; probe_bx = bx; }
            }
            cudaFree(dpp);
            cudaFree(dzz);
        }
        std::printf("\n== %s: type %d, %d -> %d, %.2f MB, %d cold copies; probe %.2f us (%.0f GB/s, %d blocks)\n", sh.name,
                    sh.type, sh.n_in, sh.n_out, bytes / 1e6, ncopy, probe_us, bytes / probe_us / 1e3, probe_bx);
        for (int T = t_lo; T <= t_hi; ++T) {
            std::vector<float> hx((size_t) T * sh.n_in);
            for (auto& v : hx) v = nd(rng);
            float *dx, *dy;
            void *dq, *dil;
            ck(cudaMalloc((void**) &dx, hx.size() * 4), "malloc");
            ck(cudaMalloc(&dq, K::native_q8_1_bytes(sh.n_in, T)), "malloc");
            ck(cudaMalloc(&dil, std::max(K::native_q8_1_il_bytes(sh.n_in, std::max(T, 2)), K::native_q8_1_bytes(sh.n_in, T))), "malloc");
            ck(cudaMalloc((void**) &dy, (size_t) T * sh.n_out * 4), "malloc");
            cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
            if (T >= 2) K::native_quantize_q8_1_il(dx, dq, dil, sh.n_in, T, s);
            else K::native_quantize_q8_1(dx, dq, sh.n_in, 1, s);
            ck(cudaStreamSynchronize(s), "quantize");
            auto run = [&](const uint8_t* w) { K::native_mmvq_il(sh.type, w, dq, dil, dy, sh.n_in, sh.n_out, T, s); };
            auto timeit = [&]() {
                float best = 1e30f;
                for (int round = 0; round < 3; ++round) {
                    for (int i = 0; i < 4; ++i) run(dw[i % ncopy]);
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < iters; ++i) run(dw[i % ncopy]);
                    cudaEventRecord(e1, s);
                    ck(cudaEventSynchronize(e1), "time");
                    float ms = 0.f;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best = std::min(best, ms * 1000.f / iters);
                }
                return best;
            };
            // today's: the table, the plain kernel
            K::native_mmvq_il_pf(0);
            K::native_mmvq_il_tune(0);
            std::vector<std::vector<float>> ref(ncopy, std::vector<float>((size_t) T * sh.n_out));
            for (int c = 0; c < std::min(ncopy, 2); ++c) {
                run(dw[c]);
                ck(cudaStreamSynchronize(s), "ref");
                cudaMemcpy(ref[c].data(), dy, ref[c].size() * 4, cudaMemcpyDeviceToHost);
            }
            const float base = timeit();
            std::printf("T=%d  today (il table rows %d)        %7.2f us  %4.0f GB/s\n", T,
                        K::native_mmvq_il_supported(sh.type, T, sh.n_out) ? 1 : 0, base, bytes / base / 1e3);
            std::vector<float> got((size_t) T * sh.n_out);
            for (int pf : pfs) {
                for (int r : rowsv) {
                    if (T > 4 && r == 4) continue;
                    K::native_mmvq_il_pf(pf);
                    K::native_mmvq_il_tune(r);
                    bool same = true;
                    for (int c = 0; c < std::min(ncopy, 2); ++c) {
                        cudaMemset(dy, 0xff, got.size() * 4);
                        run(dw[c]);
                        ck(cudaStreamSynchronize(s), "variant");
                        cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
                        same = same && std::memcmp(got.data(), ref[c].data(), got.size() * 4) == 0;
                    }
                    ++cases;
                    if (!same) ++bad;
                    const float us = timeit();
                    std::printf("T=%d  pf %d rows %d  %-6s           %7.2f us  %4.0f GB/s  %+6.2f us\n", T, pf, r,
                                same ? "same" : "DIFF", us, bytes / us / 1e3, us - base);
                }
            }
            // native_mmvq's own kernels (the plain q8_1): the exact multi-column layout (b6 0; STRATA_NO_MMVQ_ROWS2=1 makes
            // it one row a block) and B6's rows a block (1 = 2 rows, 4, 8)
            K::native_mmvq_il_pf(0);
            K::native_mmvq_il_tune(0);
            for (int b6 : b6v) {
                if (T == 1) break;
                K::native_mmvq_set_b6_rows(b6);
                auto runn = [&](const uint8_t* w) { K::native_mmvq(sh.type, w, dq, dy, sh.n_in, sh.n_out, T, s); };
                bool same = true;
                for (int c = 0; c < std::min(ncopy, 2); ++c) {
                    cudaMemset(dy, 0xff, got.size() * 4);
                    runn(dw[c]);
                    ck(cudaStreamSynchronize(s), "native");
                    cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost);
                    same = same && std::memcmp(got.data(), ref[c].data(), got.size() * 4) == 0;
                }
                ++cases;
                if (!same) ++bad;
                float best = 1e30f;
                for (int round = 0; round < 3; ++round) {
                    for (int i = 0; i < 4; ++i) runn(dw[i % ncopy]);
                    cudaEventRecord(e0, s);
                    for (int i = 0; i < iters; ++i) runn(dw[i % ncopy]);
                    cudaEventRecord(e1, s);
                    ck(cudaEventSynchronize(e1), "time");
                    float ms = 0.f;
                    cudaEventElapsedTime(&ms, e0, e1);
                    best = std::min(best, ms * 1000.f / iters);
                }
                std::printf("T=%d  native b6 %d   %-6s           %7.2f us  %4.0f GB/s  %+6.2f us\n", T, b6, same ? "same" : "DIFF",
                            best, bytes / best / 1e3, best - base);
            }
            K::native_mmvq_set_b6_rows(1);
            if (pair) {   // STRATA_QL_MMVQ_PAIR: two matrices of this shape (copies c and c + 1: keys, values) in one launch
                float* dy2;
                ck(cudaMalloc((void**) &dy2, (size_t) T * sh.n_out * 4), "malloc");
                auto two = [&](int i) {
                    run(dw[i % ncopy]);
                    K::native_mmvq_il(sh.type, dw[(i + 1) % ncopy], dq, dil, dy2, sh.n_in, sh.n_out, T, s);
                };
                auto one = [&](int i) {
                    if (!K::native_mmvq_pair_kq(sh.type, dw[i % ncopy], dw[(i + 1) % ncopy], dq, dy, dy2, sh.n_in, sh.n_out, T, s)) {
                        std::printf("native_mmvq_pair_kq refused\n");
                        std::exit(3);
                    }
                };
                std::vector<float> r1(got.size()), r2(got.size()), g1(got.size()), g2(got.size());
                two(0);
                ck(cudaStreamSynchronize(s), "two");
                cudaMemcpy(r1.data(), dy, r1.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(r2.data(), dy2, r2.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemset(dy, 0xff, got.size() * 4);
                cudaMemset(dy2, 0xff, got.size() * 4);
                one(0);
                ck(cudaStreamSynchronize(s), "one");
                cudaMemcpy(g1.data(), dy, g1.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(g2.data(), dy2, g2.size() * 4, cudaMemcpyDeviceToHost);
                const bool same = std::memcmp(r1.data(), g1.data(), r1.size() * 4) == 0 &&
                                  std::memcmp(r2.data(), g2.data(), r2.size() * 4) == 0;
                ++cases;
                if (!same) ++bad;
                auto tm = [&](auto&& f) {
                    float best = 1e30f;
                    for (int round = 0; round < 3; ++round) {
                        for (int i = 0; i < 4; ++i) f(2 * i);
                        cudaEventRecord(e0, s);
                        for (int i = 0; i < iters; ++i) f(2 * i);
                        cudaEventRecord(e1, s);
                        ck(cudaEventSynchronize(e1), "time");
                        float ms = 0.f;
                        cudaEventElapsedTime(&ms, e0, e1);
                        best = std::min(best, ms * 1000.f / iters);
                    }
                    return best;
                };
                const float t2 = tm(two), t1 = tm(one);
                std::printf("T=%d  pair: two launches %6.2f us, one launch %6.2f us  %s  %+6.2f us\n", T, t2, t1,
                            same ? "same" : "DIFF", t1 - t2);
                cudaFree(dy2);
            }
            cudaFree(dx);
            cudaFree(dq);
            cudaFree(dil);
            cudaFree(dy);
        }
        for (auto* p : dw) cudaFree(p);
    }
    if (only.empty() || only == "router") {   // the MoE router (bf16 512 x 2560): STRATA_MMVF_ROWS is read once, so its
        // A/B is two runs of this process; the outputs' FNV hash per T is printed to compare them
        const int NI = 2560, NO = 512;
        const size_t bytes = (size_t) NI * NO * 2;
        const int ncopy = std::max(1, (int) ((size_t) cold_mb * 1000000 / bytes) + 1);
        std::vector<uint16_t> hw((size_t) NI * NO);
        std::vector<uint16_t*> dw(ncopy);
        for (int c = 0; c < ncopy; ++c) {
            for (auto& v : hw) { const float f = 0.02f * nd(rng); uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t) (u >> 16); }
            ck(cudaMalloc((void**) &dw[c], bytes), "malloc");
            cudaMemcpy(dw[c], hw.data(), bytes, cudaMemcpyHostToDevice);
        }
        const char* env = std::getenv("STRATA_MMVF_ROWS");
        std::printf("\n== router: bf16 %d -> %d, %.2f MB, %d cold copies, STRATA_MMVF_ROWS=%s\n", NI, NO, bytes / 1e6, ncopy,
                    env ? env : "(unset)");
        for (int T = t_lo; T <= t_hi; ++T) {
            std::vector<float> hx((size_t) T * NI);
            for (auto& v : hx) v = nd(rng);
            float *dx, *dy;
            ck(cudaMalloc((void**) &dx, hx.size() * 4), "malloc");
            ck(cudaMalloc((void**) &dy, (size_t) T * NO * 4), "malloc");
            cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
            auto run = [&](int i) { K::bf16_gemv_fp32_mmvf_multi(dx, NI, dw[i % ncopy], dy, NO, NI, NO, T, s); };
            run(0);
            ck(cudaStreamSynchronize(s), "router");
            std::vector<uint8_t> out((size_t) T * NO * 4);
            cudaMemcpy(out.data(), dy, out.size(), cudaMemcpyDeviceToHost);
            uint64_t h = 1469598103934665603ull;
            for (uint8_t b : out) h = (h ^ b) * 1099511628211ull;
            float best = 1e30f;
            for (int round = 0; round < 3; ++round) {
                for (int i = 0; i < 4; ++i) run(i);
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) run(i);
                cudaEventRecord(e1, s);
                ck(cudaEventSynchronize(e1), "time");
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms * 1000.f / iters);
            }
            std::printf("T=%d  router %7.2f us  %4.0f GB/s  hash %016llx\n", T, best, bytes / best / 1e3, (unsigned long long) h);
            cudaFree(dx);
            cudaFree(dy);
        }
        for (auto* p : dw) cudaFree(p);
    }
    std::printf("\nmmvq_t_bench: %d variant cases, %d differ\n", cases, bad);
    return bad ? 1 : 0;
}
