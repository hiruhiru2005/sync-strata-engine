// hc_i8_bench - QuantLab (11/13) STRATA_QL_HC_BW: the int8 hyper-connection read (STRATA_QL_HC_I8, gr_down_i8 +
// gr_up_i8) against its hoisted (=1, up warps a block 1 / 2 / 4 / 8) and one-kernel (=2) forms at the model's shapes
// (n_embd 2560, hc 4, hc_lr 320; random weights repacked by fused_gr_q8_split, the inject rows bf16): every output
// (R in place, rs, inject, mixed) compared byte for byte with the old read for T tokens, with and without the pending
// write and the inject rows, then each form's time per read with the weights rotating over `--cold` copies (a layer's
// weights are cold in the engine): us a read (events, the mean of --iters, the best of 3 rounds), GB/s of the 7.0 MB
// read, and the GPU stamps' median split (down / up, a stamp kernel each; =2 puts all of it under "down").
//
//     build/hc_i8_bench [--t A B] [--iters n] [--cold n] [--mode m] [--upw w] [--no-stamps] [--seed s] [--probe]
// --probe: only the read-only yardstick (ql_bw_probe) on the same bytes; --seed: the random weights' and inputs' seed.
// --mode m: only the old read and form m (1 .. 6, see fused_gr.cu's STRATA_QL_HC_BW); --upw w: =1 with w warps an up block only.  Exit 1 when any form
// differs; a form that does not finish in 5 s (=2 waits on a counter) ends the process (exit 2).
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace strata::kernels {   // QuantLab (11/12) iq_kernels.cu: the bandwidth yardstick (tq_mix_parity --bw)
void ql_bw_probe(const unsigned long long* d_ptrs, const unsigned long long* d_sizes, int n, int blocks_x, int unroll,
                 unsigned* d_out, cudaStream_t s);
}

namespace {
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
// Q8_0 (d = amax / 127 as fp16, q = round(x / d)) of bf16 values, as fused_gr_bench's
std::vector<uint8_t> q8_0_of_bf16(const std::vector<uint16_t>& w) {
    std::vector<uint8_t> out(w.size() / 32 * 34);
    for (size_t b = 0; b < w.size() / 32; ++b) {
        float v[32], amax = 0.f;
        for (int i = 0; i < 32; ++i) {
            uint32_t u = (uint32_t) w[b * 32 + i] << 16;
            std::memcpy(&v[i], &u, 4);
            amax = std::fmax(amax, std::fabs(v[i]));
        }
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        uint32_t x;
        std::memcpy(&x, &d, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        const int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
        const uint32_t m = x & 0x7fffffu;
        uint16_t h;
        if (e <= 0) h = (uint16_t) sign;
        else if (e >= 31) h = (uint16_t) (sign | 0x7c00u);
        else h = (uint16_t) (sign | ((uint32_t) e << 10) | (m >> 13));
        uint8_t* o = &out[b * 34];
        std::memcpy(o, &h, 2);
        for (int i = 0; i < 32; ++i) o[2 + i] = (uint8_t) (int8_t) std::lround(v[i] * id);
    }
    return out;
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("hc_i8_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(3);
    }
}
// a hung =2 read (its up blocks wait on a counter) must not keep the card: give up after 5 s
void sync_or_die(cudaStream_t s, const char* what) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t e = cudaStreamQuery(s);
        if (e == cudaSuccess) return;
        if (e != cudaErrorNotReady) ck(e, what);
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            std::printf("hc_i8_bench: %s did not finish in 5 s (hang?)\n", what);
            std::exit(2);
        }
    }
}
struct Form { int mode, upw; };
}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int t_lo = 1, t_hi = 4, iters = 300, cold = 8, only_mode = -1, only_upw = 0, seed = 7;
    bool stamps = true, probe = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--t" && i + 2 < argc) { t_lo = std::atoi(argv[++i]); t_hi = std::atoi(argv[++i]); }
        else if (a == "--iters" && i + 1 < argc) iters = std::max(10, std::atoi(argv[++i]));
        else if (a == "--cold" && i + 1 < argc) cold = std::max(1, std::atoi(argv[++i]));
        else if (a == "--mode" && i + 1 < argc) only_mode = std::atoi(argv[++i]);
        else if (a == "--upw" && i + 1 < argc) only_upw = std::atoi(argv[++i]);
        else if (a == "--no-stamps") stamps = false;
        else if (a == "--probe") probe = true;
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { std::printf("hc_i8_bench: unknown argument %s\n", a.c_str()); return 3; }
    }
    t_lo = std::max(1, std::min(K::kFusedGrMaxT, t_lo));
    t_hi = std::max(t_lo, std::min(K::kFusedGrMaxT, t_hi));
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("hc_i8_bench: no CUDA device, skipped\n");
        return 77;
    }
    std::vector<Form> forms = {{0, 1}};   // forms[0]: the old read (STRATA_QL_HC_BW off)
    for (int w : {1, 2, 4, 8})
        if ((only_mode < 0 || only_mode == 1) && (only_upw == 0 || only_upw == w)) forms.push_back({1, w});
    if (only_mode < 0 || only_mode == 2) forms.push_back({2, 0});
    if (only_mode < 0 || only_mode == 3) forms.push_back({3, 0});
    for (int md : {4, 5, 6})
        if (only_mode < 0 || only_mode == md) forms.push_back({md, 0});

    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng((unsigned) seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    const std::vector<uint8_t> qd = q8_0_of_bf16(wd), qu = q8_0_of_bf16(wu);

    float *dR, *dbo, *dinj, *dxn, *dlo, *drs, *dio, *dmix;
    ck(cudaMalloc((void**) &dR, R.size() * 4), "malloc");
    ck(cudaMalloc((void**) &dbo, bo.size() * 4), "malloc");
    ck(cudaMalloc((void**) &dinj, inj.size() * 4), "malloc");
    ck(cudaMalloc((void**) &dxn, (size_t) TM * D * 4), "malloc");
    ck(cudaMalloc((void**) &dlo, (size_t) TM * LR * 4), "malloc");
    ck(cudaMalloc((void**) &drs, (size_t) TM * HC * 4), "malloc");
    ck(cudaMalloc((void**) &dio, (size_t) TM * HC * 4), "malloc");
    ck(cudaMalloc((void**) &dmix, (size_t) TM * N * 4), "malloc");
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    // `cold` copies of the layer: split Q8_0 down / up, bf16 inject, w_norm (7.1 MB each)
    std::vector<uint8_t*> cqd(cold), cqu(cold);
    std::vector<uint16_t*> cwi(cold);
    std::vector<float*> cwn(cold);
    for (int c = 0; c < cold; ++c) {
        ck(cudaMalloc((void**) &cqd[c], qd.size()), "malloc");
        ck(cudaMalloc((void**) &cqu[c], qu.size()), "malloc");
        ck(cudaMalloc((void**) &cwi[c], wi.size() * 2), "malloc");
        ck(cudaMalloc((void**) &cwn[c], wn.size() * 4), "malloc");
        cudaMemcpy(cqd[c], qd.data(), qd.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(cqu[c], qu.data(), qu.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(cwi[c], wi.data(), wi.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(cwn[c], wn.data(), wn.size() * 4, cudaMemcpyHostToDevice);
        if (!K::fused_gr_q8_split(cqd[c], (int64_t) (qd.size() / 34), s) || !K::fused_gr_q8_split(cqu[c], (int64_t) (qu.size() / 34), s)) {
            std::printf("hc_i8_bench: fused_gr_q8_split failed\n");
            return 3;
        }
    }
    ck(cudaStreamSynchronize(s), "split");
    K::fused_gr_set_q8_i8(1);
    if (probe) {   // --probe: only reading the same bytes (down's 3.48 MB codes + scales, up's), rotating over the copies
        unsigned long long *dp, *dsz;
        unsigned* dout;
        ck(cudaMalloc((void**) &dp, (size_t) 2 * cold * 8), "malloc");
        ck(cudaMalloc((void**) &dsz, (size_t) 2 * cold * 8), "malloc");
        ck(cudaMalloc((void**) &dout, 4), "malloc");
        std::vector<unsigned long long> hp(2 * cold), hs(2 * cold);
        for (int c = 0; c < cold; ++c) {
            hp[2 * c] = (unsigned long long) cqd[c]; hs[2 * c] = qd.size();
            hp[2 * c + 1] = (unsigned long long) cqu[c]; hs[2 * c + 1] = qu.size();
        }
        cudaMemcpy(dp, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dsz, hs.data(), hs.size() * 8, cudaMemcpyHostToDevice);
        cudaEvent_t p0, p1;
        cudaEventCreate(&p0);
        cudaEventCreate(&p1);
        for (int unroll : {1, 4})
            for (int bx : {68, 136, 220, 272, 544, 850, 1088}) {
                double best = 1e30;
                for (int round = 0; round < 3; ++round) {
                    cudaEventRecord(p0, s);
                    for (int i = 0; i < iters; ++i) K::ql_bw_probe(dp + (i % (2 * cold)), dsz + (i % (2 * cold)), 1, bx, unroll, dout, s);
                    cudaEventRecord(p1, s);
                    sync_or_die(s, "probe");
                    float ms = 0;
                    cudaEventElapsedTime(&ms, p0, p1);
                    best = std::min(best, 1e3 * ms / iters);
                }
                const double mb = (qd.size() + qu.size()) / 2.0;
                std::printf("PROBE read-only %.2f MB, %4d blocks of 256, unroll %d | %6.2f us a launch (%4.0f GB/s)\n", mb / 1e6, bx, unroll,
                            best, mb / (best * 1e3));
            }
        for (int bx : {136, 220, 300, 544}) {   // down + up of one copy in ONE launch (gridDim.y 2): one 7 MB read
            double best = 1e30;
            for (int round = 0; round < 3; ++round) {
                cudaEventRecord(p0, s);
                for (int i = 0; i < iters; ++i) K::ql_bw_probe(dp + 2 * (i % cold), dsz + 2 * (i % cold), 2, bx, 4, dout, s);
                cudaEventRecord(p1, s);
                sync_or_die(s, "probe");
                float ms = 0;
                cudaEventElapsedTime(&ms, p0, p1);
                best = std::min(best, 1e3 * ms / iters);
            }
            const double mb = (double) (qd.size() + qu.size());
            std::printf("PROBE read-only %.2f MB in one launch, %4d x 2 blocks of 256 | %6.2f us a launch (%4.0f GB/s)\n", mb / 1e6, bx, best,
                        mb / (best * 1e3));
        }
        return 0;
    }
    auto args = [&](std::vector<K::FusedGrArgs>& as, int T, int apply, int inject) {
        as.assign((size_t) cold * TM, K::FusedGrArgs());
        for (int c = 0; c < cold; ++c)
            for (int t = 0; t < T; ++t) {
                K::FusedGrArgs& a = as[(size_t) c * TM + t];
                a.R = dR + (size_t) t * D; a.R_out = dR + (size_t) t * D; a.apply = apply != 0;
                a.bo_prev = dbo + (size_t) t * N; a.inj_prev = dinj + (size_t) t * HC;
                a.w_norm = cwn[c]; a.w_inject = inject ? cwi[c] : nullptr;
                a.q8_down = cqd[c]; a.q8_up = cqu[c];
                a.lo = dlo + (size_t) t * LR; a.rs = drs + (size_t) t * HC;
                a.inject_out = dio + (size_t) t * HC; a.mixed = dmix + (size_t) t * N;
            }
    };
    const double bytes_down = (double) LR * D * 34 / 32, bytes_inj = (double) HC * D * 2, bytes_up = (double) D * LR * 34 / 32;
    unsigned long long* dst = nullptr;
    ck(cudaMalloc((void**) &dst, (size_t) iters * 4 * 8), "malloc");
    std::vector<unsigned long long> hst((size_t) iters * 4);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int failures = 0;
    std::vector<K::FusedGrArgs> as;
    // 1) bitwise: every form against the old read (copy 0 of the weights)
    for (int T = t_lo; T <= t_hi; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                args(as, T, apply, inject);
                std::vector<std::vector<float>> out(forms.size());
                for (size_t f = 0; f < forms.size(); ++f) {
                    K::fused_gr_set_hc_bw(forms[f].mode, forms[f].upw);
                    cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
                    cudaMemset(drs, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dio, 0xff, (size_t) TM * HC * 4);
                    cudaMemset(dmix, 0xff, (size_t) TM * N * 4);
                    K::fused_gr_read_multi(as.data(), T, dxn, s);
                    sync_or_die(s, "a read");
                    auto& o = out[f];
                    o.resize((size_t) T * (D + HC + HC + N));
                    float* p = o.data();
                    cudaMemcpy(p, dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost); p += (size_t) T * D;
                    cudaMemcpy(p, drs, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
                    cudaMemcpy(p, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost); p += (size_t) T * HC;
                    cudaMemcpy(p, dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost);
                }
                std::printf("T %d apply %d inject %d |", T, apply, inject);
                for (size_t f = 1; f < forms.size(); ++f) {
                    const bool same = std::memcmp(out[0].data(), out[f].data(), out[0].size() * 4) == 0;
                    if (!same) ++failures;
                    std::printf(" bw%d/w%d %s", forms[f].mode, forms[f].upw, same ? "equal" : "DIFFERS");
                    if (!same) {   // which outputs: R, rs, inject, mixed (mismatching floats)
                        const size_t nb[4] = {(size_t) T * D, (size_t) T * HC, (size_t) T * HC, (size_t) T * N};
                        size_t o0 = 0;
                        std::printf("(");
                        for (int r = 0; r < 4; ++r) {
                            size_t bad = 0;
                            size_t first = 0;
                            for (size_t q = o0; q < o0 + nb[r]; ++q)
                                if (std::memcmp(&out[0][q], &out[f][q], 4) != 0 && bad++ == 0) first = q;
                            if (bad) std::printf("[%zu: %.9g vs %.9g] ", first - o0, out[0][first], out[f][first]);
                            std::printf("%s%zu", r ? "/" : "R/rs/inj/mix ", bad);
                            o0 += nb[r];
                        }
                        std::printf(")");
                    }
                }
                std::printf("\n");
            }
    // 2) time: the engine's case (apply, inject), the weights rotating over `cold` copies
    for (int T = t_lo; T <= t_hi; ++T) {
        args(as, T, 1, 1);
        for (size_t f = 0; f < forms.size(); ++f) {
            K::fused_gr_set_hc_bw(forms[f].mode, forms[f].upw);
            cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice);   // (R grows by bo a read: harmless here)
            for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(&as[(size_t) (i % cold) * TM], T, dxn, s);
            sync_or_die(s, "warm-up");
            double best = 1e30;
            for (int round = 0; round < 3; ++round) {
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(&as[(size_t) (i % cold) * TM], T, dxn, s);
                cudaEventRecord(e1, s);
                sync_or_die(s, "timed reads");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, 1e3 * ms / iters);
            }
            const double gb = (bytes_down + bytes_inj + bytes_up) / (best * 1e3);
            std::printf("TIME T %d bw%d/w%d | %6.2f us a read (%5.0f GB/s of %.2f MB)", T, forms[f].mode, forms[f].upw, best, gb,
                        (bytes_down + bytes_inj + bytes_up) / 1e6);
            if (stamps) {   // 0 start, 1 after down, 2 after up (gpu_stamp kernels in between)
                for (int i = 0; i < iters; ++i) {
                    K::gpu_stamp(dst, i * 4, s);
                    K::fused_gr_read_multi(&as[(size_t) (i % cold) * TM], T, dxn, s, dst, i * 4 + 1);
                    K::gpu_stamp(dst, i * 4 + 3, s);
                }
                sync_or_die(s, "stamped reads");
                cudaMemcpy(hst.data(), dst, hst.size() * 8, cudaMemcpyDeviceToHost);
                std::vector<double> dn, up;
                for (int i = 1; i < iters; ++i) {
                    const unsigned long long* p = &hst[(size_t) i * 4];
                    dn.push_back((double) (p[2] - p[1]));
                    up.push_back((double) (p[3] - p[2]));
                }
                auto med = [](std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2] / 1e3; };
                const double md = med(dn), mu = med(up);
                std::printf(" | stamps: down %5.2f us (%4.0f GB/s) up %5.2f us (%4.0f GB/s)", md, (bytes_down + bytes_inj) / (md * 1e3), mu,
                            bytes_up / (mu * 1e3));
            }
            std::printf("\n");
        }
    }
    K::fused_gr_set_hc_bw(-1, 0);
    K::fused_gr_set_q8_i8(0);
    std::printf("hc_i8_bench: %d differing\n", failures);
    return failures ? 1 : 0;
}
