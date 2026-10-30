// src/kernels/tq_mix_parity.cpp - QuantLab: a layer of per-expert trellis K (native_experts.txt v6, tq_mix.hpp) on the
// GPU against the CPU, without a model.
//
//     build/tq_mix_parity [--hadamard] [--r1q2]
//
// Twelve random experts, (K gate/up, K down) = (2, 3) .. (13, 14), each its own group of NT tokens, in ONE grouped launch
// of a mixed layer (kTqMixedType, the K pair in each group pointer's top byte), against the CPU's trellis kernels
// (tq_cpu.hpp: K 2..12 checked against QuantLab's Python reader by `tq_cpu_parity --vectors`) through the engine's
// entry points (native_quant_act -> native_gu_rows -> native_quant_h -> native_down_rows).  The GPU reads the int8
// codebook and q8_1 activations, so the bound is native_expert_parity's: relative L1 error < 3e-2 per expert.
// Then the prompt path's dequantizers (iq_dequant_gu_f16 / iq_dequant_f16) of every one of those types against a
// host decode of the same blocks (the fp16 codebook, as ggml's to_float): max |error| <= 2^-10 of the value's scale.
// QuantLab (10/31) --r1q2: the same experts as R1Q2 blobs (cpu::ql_r1q2_fmt: 4K code bytes a block, the scales after
// the codes): the mixed-layer GPU kernels' R1Q2 branch against the CPU's R1Q2 rows, the prompt path's
// iq_dequant_expert_r1q2_f16 against the host decode with d = colscale * p2[q] * rowbase, and the PCIe staging copy
// (fetch_blobs_mixed with the R1Q2 scale bytes) copying every blob's own bytes.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/tq_mix.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "ggml-common-tq.h"   // the fp16 codebook, host data here

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace k = strata::kernels;

namespace {
constexpr int NT = 3, NE = 12;   // 10/26: (2,3) .. (13,14)
constexpr int64_t H = 2560, FF = 640;

int tq_type(int K) {
    switch (K) {
        case 8: return 144;
        case 6: return 145;
        case 7: return 146;
        case 5: return 150;
        case 4: return 151;
        default: return cpu::kTqPseudoBase + K;
    }
}

float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {   // subnormal
            float v = std::ldexp((float) m, -24);
            return (h >> 15) ? -v : v;
        }
    } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else bits = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
uint16_t f2h(float v) {   // round to nearest (values here are normal and small)
    uint32_t b;
    std::memcpy(&b, &v, 4);
    const uint32_t s = (b >> 16) & 0x8000u;
    const int e = (int) ((b >> 23) & 255) - 127 + 15;
    uint32_t m = (b >> 13) & 1023;
    if ((b >> 12) & 1) ++m;
    if (m == 1024) return (uint16_t) (s | ((e + 1) << 10));
    return (uint16_t) (s | (e << 10) | m);
}

// One 128-value block decoded on the host: QuantLab's tools/tq.py (and ggml-common-tq.h's comment) in C++.
void decode_codes(const uint8_t* qs, int K, float d, float* y);
void decode_block(const uint8_t* b, int K, float* y) {
    uint16_t dh;
    std::memcpy(&dh, b, 2);
    decode_codes(b + 2, K, h2f(dh), y);
}
// the block's codes alone (R1Q2 blocks: 4K bytes, the scale d given)
void decode_codes(const uint8_t* qs, int K, float d, float* y) {
    for (int t = 0; t < 32; ++t) {
        uint32_t s;
        if (K == 8) s = ((uint32_t) qs[(t + 31) % 32] << 8) | qs[t];
        else {
            s = 0;
            const int nb = 32 * K;
            for (int j = 0; j < 15; ++j) {
                const int bit = ((31 - t) * K + j) % nb;
                s |= (uint32_t) ((qs[bit >> 3] >> (bit & 7)) & 1) << j;
            }
        }
        const uint32_t x = (s & TQ_STATE_MASK) * 0x9e3779b1u;
        const uint32_t a = x >> 21, c = (x >> 10) & 2047u;
        y[4 * t + 0] = d * h2f(tq_lut_f16[2 * a]);
        y[4 * t + 1] = d * h2f(tq_lut_f16[2 * a + 1]);
        y[4 * t + 2] = d * h2f(tq_lut_f16[2 * c]);
        y[4 * t + 3] = d * h2f(tq_lut_f16[2 * c + 1]);
    }
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return false;
}

double rel_l1(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { num += std::fabs((double) a[i] - b[i]); den += std::fabs((double) b[i]); }
    return num / (den + 1e-30);
}
}  // namespace

// QuantLab (10/31): an R1Q2 matrix's block scale (m 0 gate, 1 up, 2 down; r the row, b the column block)
float r1q2_d(const cpu::NativeFmt& f, const uint8_t* blob, int m, int64_t r, int64_t b) {
    const int64_t rows = m < 2 ? f.n_ff : f.n_embd, nb = (m < 2 ? f.n_embd : f.n_ff) / 128, qb = (nb + 3) / 4;
    const uint8_t* sc = blob + f.sc_off + cpu::ql_r1q2_matrix_off(f, m);
    uint16_t rb, cs;
    std::memcpy(&rb, sc + 2 * r, 2);
    std::memcpy(&cs, sc + 2 * rows + 2 * b, 2);
    const int q = (sc[2 * rows + 2 * nb + r * qb + b / 4] >> (2 * (b % 4))) & 3;
    return h2f(cs) * f.p2[m][q] * h2f(rb);
}

int main(int argc, char** argv) {
    bool had = false, r1q2 = false;
    for (int i = 1; i < argc; ++i) {
        had = had || std::string(argv[i]) == "--hadamard";
        r1q2 = r1q2 || std::string(argv[i]) == "--r1q2";
    }
    std::printf("tq_mix_parity: %d experts of per-expert trellis K in one grouped launch%s%s\n", NE,
                had ? " (Hadamard-folded down: identity signs, block 128)" : "", r1q2 ? " (R1Q2 blobs)" : "");
    // R1Q2's per-matrix table p2[q] = 2^(lo + q * step) (the SAR1Q2 layers' lo ~ -0.25, step ~ 0.17)
    float P2[3][4];
    for (int m = 0; m < 3; ++m)
        for (int q = 0; q < 4; ++q) P2[m][q] = std::exp2(-0.25f + 0.02f * m + q * (0.17f - 0.01f * m));
    int failures = 0;
    // the experts: random stream bytes (every state reachable) and sane fp16 scales at every block start
    std::vector<cpu::NativeFmt> fmt(NE);
    std::vector<std::vector<uint8_t>> blob(NE);
    std::vector<int> kg(NE), kd(NE);
    std::mt19937 rng(20261025);
    std::uniform_real_distribution<float> sd(0.004f, 0.02f);
    for (int e = 0; e < NE; ++e) {
        kg[e] = 2 + e;
        kd[e] = 3 + e;
        std::string err;
        if (!cpu::native_fmt(tq_type(kg[e]), tq_type(kd[e]), H, FF, fmt[e], err)) {
            std::printf("native_fmt K %d/%d: %s  FAIL\n", kg[e], kd[e], err.c_str());
            return 1;
        }
        if (had) { fmt[e].had_block = 128; fmt[e].had_d = true; fmt[e].d_signs = nullptr; }
        if (r1q2 && !cpu::ql_r1q2_fmt(fmt[e], P2, err)) {
            std::printf("ql_r1q2_fmt K %d/%d: %s  FAIL\n", kg[e], kd[e], err.c_str());
            return 1;
        }
        blob[e].resize(fmt[e].bytes);
        for (auto& v : blob[e]) v = (uint8_t) rng();
        if (r1q2) {   // the scales: rowbase 0.5 .. 2, colscale 0.004 .. 0.02 (positive; R1Q2 has no sign), q random
            std::uniform_real_distribution<float> rbd(0.5f, 2.0f);
            for (int m = 0; m < 3; ++m) {
                const int64_t rows = m < 2 ? FF : H, nb = (m < 2 ? H : FF) / 128;
                uint8_t* sc = blob[e].data() + fmt[e].sc_off + cpu::ql_r1q2_matrix_off(fmt[e], m);
                for (int64_t r = 0; r < rows; ++r) { const uint16_t h = f2h(rbd(rng)); std::memcpy(sc + 2 * r, &h, 2); }
                for (int64_t b = 0; b < nb; ++b) { const uint16_t h = f2h(sd(rng)); std::memcpy(sc + 2 * rows + 2 * b, &h, 2); }
            }
            continue;
        }
        const size_t bg = 2 + 4 * (size_t) kg[e], bd = 2 + 4 * (size_t) kd[e];
        for (size_t o = 0; o + bg <= fmt[e].down_off; o += bg) {
            const uint16_t d = f2h(rng() & 1 ? sd(rng) : -sd(rng));
            std::memcpy(blob[e].data() + o, &d, 2);
        }
        for (size_t o = fmt[e].down_off; o + bd <= fmt[e].bytes; o += bd) {
            const uint16_t d = f2h(rng() & 1 ? sd(rng) : -sd(rng));
            std::memcpy(blob[e].data() + o, &d, 2);
        }
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);

    // (a) the CPU, per expert
    std::vector<float> want((size_t) NE * NT * H);
    for (int e = 0; e < NE; ++e) {
        const cpu::NativeFmt& f = fmt[e];
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
        const void* a[NT];
        float* fp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int t = 0; t < NT; ++t) {
            cpu::native_quant_act(f, x.data() + t * H, act[t].data());
            a[t] = act[t].data();
            fp[t] = ff[t].data();
        }
        cpu::native_gu_rows(f, blob[e].data(), a, NT, fp, 0, (int) FF);
        for (int t = 0; t < NT; ++t) {
            cpu::native_quant_h(f, ff[t].data(), hq[t].data());   // (rotates h when the down is folded)
            hp[t] = hq[t].data();
            op[t] = want.data() + ((size_t) e * NT + t) * H;
        }
        cpu::native_down_rows(f, blob[e].data(), hp, NT, op, 0, (int) H);
    }

    // (b) the GPU: the experts packed like a mixed layer in experts.bin, one group each, tagged pointers
    std::vector<size_t> off(NE);
    size_t total = 0;
    for (int e = 0; e < NE; ++e) { off[e] = total; total += blob[e].size(); }
    uint8_t* d_arena = nullptr;
    float *d_x = nullptr, *d_out = nullptr;
    void *d_xq = nullptr, *d_scr = nullptr;
    unsigned long long* d_ptr = nullptr;
    int32_t *d_start = nullptr, *d_n = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    const int64_t cap_e = (int64_t) NE * NT;
    cudaStream_t s = nullptr;
    if (!ck(cudaStreamCreate(&s), "stream") || !ck(cudaMalloc((void**) &d_arena, total), "arena") ||
        !ck(cudaMalloc((void**) &d_x, sizeof(float) * NT * H), "x") ||
        !ck(cudaMalloc(&d_xq, (size_t) NT * (H / 32) * 36), "xq") ||
        !ck(cudaMalloc(&d_scr, k::native_expert_scratch_bytes(cap_e, FF)), "scratch") ||
        !ck(cudaMalloc((void**) &d_out, sizeof(float) * cap_e * H), "out") ||
        !ck(cudaMalloc((void**) &d_ptr, sizeof(unsigned long long) * NE), "ptr") ||
        !ck(cudaMalloc((void**) &d_start, sizeof(int32_t) * (NE + 1)), "start") ||
        !ck(cudaMalloc((void**) &d_n, sizeof(int32_t)), "n") ||
        !ck(cudaMalloc((void**) &d_dst, sizeof(int32_t) * cap_e), "dst") ||
        !ck(cudaMalloc((void**) &d_tok, sizeof(int32_t) * cap_e), "tok"))
        return 1;
    std::vector<unsigned long long> ptr(NE);
    std::vector<int32_t> start(NE + 1), dst(cap_e), tok(cap_e);
    for (int e = 0; e < NE; ++e) {
        if (!ck(cudaMemcpy(d_arena + off[e], blob[e].data(), blob[e].size(), cudaMemcpyHostToDevice), "upload")) return 1;
        ptr[e] = (unsigned long long) (d_arena + off[e]);
        if ((ptr[e] & ~k::kTqAddrMask) != 0) { std::printf("device address 0x%llx has its top byte in use  FAIL\n", ptr[e]); return 1; }
        ptr[e] |= k::tq_mix_tag(kg[e], kd[e]);
        start[e] = e * NT;
        for (int t = 0; t < NT; ++t) { dst[e * NT + t] = e * NT + t; tok[e * NT + t] = t; }
    }
    start[NE] = (int32_t) cap_e;
    const int32_t ng = NE;
    if (!ck(cudaMemcpy(d_x, x.data(), sizeof(float) * NT * H, cudaMemcpyHostToDevice), "x up") ||
        !ck(cudaMemcpy(d_ptr, ptr.data(), sizeof(unsigned long long) * NE, cudaMemcpyHostToDevice), "ptr up") ||
        !ck(cudaMemcpy(d_start, start.data(), sizeof(int32_t) * (NE + 1), cudaMemcpyHostToDevice), "start up") ||
        !ck(cudaMemcpy(d_n, &ng, sizeof(int32_t), cudaMemcpyHostToDevice), "n up") ||
        !ck(cudaMemcpy(d_dst, dst.data(), sizeof(int32_t) * cap_e, cudaMemcpyHostToDevice), "dst up") ||
        !ck(cudaMemcpy(d_tok, tok.data(), sizeof(int32_t) * cap_e, cudaMemcpyHostToDevice), "tok up"))
        return 1;
    k::quantize_q8_1_rows(d_x, NT, H, d_xq, s);
    k::NativeExpertLayout L = k::native_expert_layout(145, 146, H, FF);   // the geometry; the K come from the tags
    L.gu_type = L.d_type = k::kTqMixedType;
    if (had) L.had_block = 128;   // d_signs nullptr: identity signs, as the CPU side
    if (r1q2) {
        L.r1q2 = true;
        for (int m = 0; m < 3; ++m)
            for (int q = 0; q < 4; ++q) L.p2[m][q] = P2[m][q];
    }
    for (int64_t gy : {(int64_t) 0, (int64_t) 3}) {   // one block row per group, and a strided launch
        if (!ck(cudaMemsetAsync(d_out, 0, sizeof(float) * cap_e * H, s), "zero")) return 1;
        k::native_expert_grouped(L, d_ptr, d_start, d_n, d_dst, d_tok, NE, cap_e, d_xq, d_scr, d_out, s, gy);
        if (!ck(cudaStreamSynchronize(s), "grouped")) return 1;
        std::vector<float> got((size_t) cap_e * H);
        if (!ck(cudaMemcpy(got.data(), d_out, sizeof(float) * cap_e * H, cudaMemcpyDeviceToHost), "out down")) return 1;
        for (int e = 0; e < NE; ++e) {
            const size_t o = (size_t) e * NT * H;
            const double r = rel_l1(got.data() + o, want.data() + o, (size_t) NT * H);
            const bool ok = r < 3e-2;
            std::printf("  grid %s: expert K %2d/%2d (types %d/%d): GPU vs CPU rel L1 %.2e  %s\n", gy ? "strided" : "per group",
                        kg[e], kd[e], fmt[e].gu_type, fmt[e].d_type, r, ok ? "ok" : "FAIL");
            failures += !ok;
        }
    }

    // (c) the prompt path's dequantizers against the host decode
    uint16_t* d_h = nullptr;
    if (!ck(cudaMalloc((void**) &d_h, sizeof(uint16_t) * 2 * FF * H), "dq")) return 1;
    std::vector<uint16_t> hq16((size_t) 2 * FF * H);
    if (r1q2) {
        uint16_t* d_h2 = nullptr;
        if (!ck(cudaMalloc((void**) &d_h2, sizeof(uint16_t) * H * FF), "dq2")) return 1;
        std::vector<uint16_t> hd16((size_t) H * FF);
        for (int e = 0; e < NE; ++e) {
            const cpu::NativeFmt& f = fmt[e];
            k::iq_dequant_expert_r1q2_f16(f.gu_type, f.d_type, d_arena + off[e], FF, H, f.up_off, f.down_off, f.sc_off, f.p2,
                                          d_h, d_h2, s);
            if (!ck(cudaStreamSynchronize(s), "dq r1q2") ||
                !ck(cudaMemcpy(hq16.data(), d_h, sizeof(uint16_t) * 2 * FF * H, cudaMemcpyDeviceToHost), "dq gu down") ||
                !ck(cudaMemcpy(hd16.data(), d_h2, sizeof(uint16_t) * H * FF, cudaMemcpyDeviceToHost), "dq d down"))
                return 1;
            std::vector<float> ref(128);
            double worst = 0, scale = 0, worst_d = 0, scale_d = 0;
            const size_t cg = 4 * (size_t) kg[e], cd = 4 * (size_t) kd[e];
            for (int64_t r = 0; r < FF; ++r)
                for (int part = 0; part < 2; ++part)
                    for (int64_t b = 0; b < H / 128; ++b) {
                        decode_codes(blob[e].data() + (part ? f.up_off : 0) + (size_t) r * f.gu_row + b * cg, kg[e],
                                     r1q2_d(f, blob[e].data(), part, r, b), ref.data());
                        const uint16_t* g = hq16.data() + ((size_t) (2 * r + part) * H + b * 128);
                        for (int i = 0; i < 128; ++i) {
                            worst = std::max(worst, std::fabs((double) h2f(g[i]) - ref[i]));
                            scale = std::max(scale, std::fabs((double) ref[i]));
                        }
                    }
            for (int64_t r = 0; r < H; ++r)
                for (int64_t b = 0; b < FF / 128; ++b) {
                    decode_codes(blob[e].data() + f.down_off + (size_t) r * f.d_row + b * cd, kd[e],
                                 r1q2_d(f, blob[e].data(), 2, r, b), ref.data());
                    const uint16_t* g = hd16.data() + ((size_t) r * FF + b * 128);
                    for (int i = 0; i < 128; ++i) {
                        worst_d = std::max(worst_d, std::fabs((double) h2f(g[i]) - ref[i]));
                        scale_d = std::max(scale_d, std::fabs((double) ref[i]));
                    }
                }
            const bool ok = worst <= scale / 1024 && worst_d <= scale_d / 1024;
            std::printf("  prompt dequant R1Q2 K %2d/%2d: gate/up max |err| %.2e of %.2e, down %.2e of %.2e  %s\n", kg[e], kd[e],
                        worst, scale, worst_d, scale_d, ok ? "ok" : "FAIL");
            failures += !ok;
        }
        // (d) the PCIe share's staging copy: each blob's own bytes (from its tag and the R1Q2 scale bytes) to its slot
        size_t slot = 0;
        for (int e = 0; e < NE; ++e) slot = std::max(slot, (size_t) fmt[e].bytes);
        slot = (slot + 15) / 16 * 16;
        uint8_t* d_stage = nullptr;
        int32_t* d_cnt = nullptr;
        if (!ck(cudaMalloc((void**) &d_stage, slot * NE), "stage") || !ck(cudaMalloc((void**) &d_cnt, sizeof(int32_t)), "cnt") ||
            !ck(cudaMemset(d_stage, 0xA5, slot * NE), "stage fill") ||
            !ck(cudaMemcpy(d_cnt, &ng, sizeof(int32_t), cudaMemcpyHostToDevice), "cnt up"))
            return 1;
        const int64_t gu_unit = 2 * FF * (H / 128), d_unit = H * (FF / 128);
        k::fetch_blobs_mixed(d_ptr, d_cnt, d_stage, (int64_t) slot, gu_unit, d_unit, NE, s,
                             (int64_t) cpu::ql_r1q2_scale_bytes(fmt[0]));
        if (!ck(cudaStreamSynchronize(s), "fetch")) return 1;
        std::vector<uint8_t> st(slot * NE);
        if (!ck(cudaMemcpy(st.data(), d_stage, slot * NE, cudaMemcpyDeviceToHost), "stage down")) return 1;
        for (int e = 0; e < NE; ++e) {
            const bool same = std::memcmp(st.data() + (size_t) e * slot, blob[e].data(), blob[e].size()) == 0;
            const bool past = blob[e].size() == slot || st[(size_t) e * slot + blob[e].size()] == 0xA5;   // nothing past its end
            std::printf("  staging copy R1Q2 K %2d/%2d: %zu bytes %s%s\n", kg[e], kd[e], blob[e].size(),
                        same ? "equal" : "DIFFERENT", past ? "" : " (wrote past its end)");
            failures += !(same && past);
        }
        std::printf("%d failures\n", failures);
        return failures ? 1 : 0;
    }
    for (int e = 0; e < NE; ++e) {
        const cpu::NativeFmt& f = fmt[e];
        const uint8_t* dev = d_arena + off[e];
        double worst = 0, scale = 0;
        // gate/up: row 2r = gate row r, row 2r + 1 = up row r (dequant_gu_kernel's interleave)
        k::iq_dequant_gu_f16(f.gu_type, dev, dev + f.up_off, FF, H, d_h, s);
        if (!ck(cudaStreamSynchronize(s), "dq gu") ||
            !ck(cudaMemcpy(hq16.data(), d_h, sizeof(uint16_t) * 2 * FF * H, cudaMemcpyDeviceToHost), "dq gu down"))
            return 1;
        std::vector<float> ref(128);
        const size_t bg = 2 + 4 * (size_t) kg[e];
        for (int64_t r = 0; r < FF; ++r)
            for (int part = 0; part < 2; ++part)
                for (int64_t b = 0; b < H / 128; ++b) {
                    decode_block(blob[e].data() + (part ? f.up_off : 0) + (size_t) r * f.gu_row + b * bg, kg[e], ref.data());
                    const uint16_t* g = hq16.data() + ((size_t) (2 * r + part) * H + b * 128);
                    for (int i = 0; i < 128; ++i) {
                        worst = std::max(worst, std::fabs((double) h2f(g[i]) - ref[i]));
                        scale = std::max(scale, std::fabs((double) ref[i]));
                    }
                }
        // down
        k::iq_dequant_f16(f.d_type, dev + f.down_off, H * FF, d_h, s);
        if (!ck(cudaStreamSynchronize(s), "dq d") ||
            !ck(cudaMemcpy(hq16.data(), d_h, sizeof(uint16_t) * H * FF, cudaMemcpyDeviceToHost), "dq d down"))
            return 1;
        const size_t bd = 2 + 4 * (size_t) kd[e];
        double worst_d = 0, scale_d = 0;
        for (int64_t r = 0; r < H; ++r)
            for (int64_t b = 0; b < FF / 128; ++b) {
                decode_block(blob[e].data() + f.down_off + (size_t) r * f.d_row + b * bd, kd[e], ref.data());
                const uint16_t* g = hq16.data() + ((size_t) r * FF + b * 128);
                for (int i = 0; i < 128; ++i) {
                    worst_d = std::max(worst_d, std::fabs((double) h2f(g[i]) - ref[i]));
                    scale_d = std::max(scale_d, std::fabs((double) ref[i]));
                }
            }
        const bool ok = worst <= scale / 1024 && worst_d <= scale_d / 1024;
        std::printf("  prompt dequant K %2d/%2d: gate/up max |err| %.2e of %.2e, down %.2e of %.2e  %s\n", kg[e], kd[e],
                    worst, scale, worst_d, scale_d, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
