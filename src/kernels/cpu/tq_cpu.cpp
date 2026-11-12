// src/kernels/cpu/tq_cpu.cpp - see include/strata/kernels/cpu/tq_cpu.hpp: the state table, the scalar reference
// kernel, the activation unpacking and the run-time ISA choice.  Compiled for the baseline x86-64 (no AVX), so
// everything here runs on any CPU; the AVX2 / AVX-512 kernels are in their own translation units.
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "tq_states.hpp"
#include "tq_computed.hpp"
#include "strata/kernels/tq_lut.hpp"   // QuantLab (11/4): the codebook (built-in, or a pack's tq_lut_f16.bin)

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// The AVX2 / AVX-512 units (tq_avx2.cpp, tq_avx512.cpp) exist on x86-64 builds only; anything else (arm64: the Metal build)
// runs the scalar kernel.
#if defined(__x86_64__) || defined(_M_X64)
#define STRATA_TQ_X86 1
#endif

namespace strata::kernels::cpu {
namespace {

alignas(64) int16_t g_table[(tq_detail::kTqStateMask + 1) * 4];
alignas(64) int16_t g_table_c[(tq_detail::kTqStateMask + 1) * 4];

void fill_table() {
    const uint16_t* lut = strata::kernels::tq_lut_f16_host();   // QuantLab (11/4): tq_lut_f16, unless a pack replaced it
    for (uint32_t s = 0; s <= tq_detail::kTqStateMask; ++s) {
        const uint32_t x = s * 0x9e3779b1u;
        const uint32_t pts[2] = {x >> 21, (x >> 10) & 2047u};   // the hash's two codebook points
        for (int h = 0; h < 2; ++h)
            for (int j = 0; j < 2; ++j) {
                // fp16 * 2^13 is exact in float; round half away from zero, as the fork's ggml-cpu table does
                const float v = tq_detail::tq_half_to_float(lut[2 * pts[h] + j]) * 8192.0f;
                g_table[4 * s + 2 * h + j] = (int16_t) (v < 0.0f ? v - 0.5f : v + 0.5f);
            }
    }
}

void fill_table_computed() {
    for (uint32_t s = 0; s <= tq_detail::kTqStateMask; ++s) tq_detail::comp_state4(s, g_table_c + 4 * s);
}

// QuantLab (11/12): the bih3 points (iq_kernels.cu tq_bih3p: T0 = [x0.b1 x0.b2 x0.b3 x1.b1], T1 = [x1.b2 x1.b3 x2.b1
// x2.b2], T2 = [x2.b3 x3.b1 x3.b2 x3.b3], weight j = (T0.bj + T1.bj + T2.bj - 384) / 128), x 2^13 = (sum - 384) * 64
alignas(64) int16_t g_table_b[(tq_detail::kTqStateMask + 1) * 4];
void fill_table_bih3() {
    auto by = [](uint32_t x, int k) { return (int) ((x >> (8 * k)) & 255u); };
    for (uint32_t s = 0; s <= tq_detail::kTqStateMask; ++s) {
        const uint32_t x0 = s * 0x9E3779B1u, x1 = s * 0x85EBCA6Bu, x2 = s * 0xC2B2AE35u, x3 = s * 0x27D4EB2Fu;
        const int t0[4] = {by(x0, 1), by(x0, 2), by(x0, 3), by(x1, 1)};
        const int t1[4] = {by(x1, 2), by(x1, 3), by(x2, 1), by(x2, 2)};
        const int t2[4] = {by(x2, 3), by(x3, 1), by(x3, 2), by(x3, 3)};
        for (int j = 0; j < 4; ++j) g_table_b[4 * s + j] = (int16_t) ((t0[j] + t1[j] + t2[j] - 384) * 64);
    }
}

int env_isa_cap() {
    const char* e = std::getenv("STRATA_TQ_ISA");
    if (e == nullptr || *e == 0) return 2;
    const std::string v(e);
    if (v == "scalar") return 0;
    if (v == "avx2") return 1;      // the plain AVX2 rows, also where AVX-VNNI exists
    if (v == "avxvnni") return 3;   // QuantLab (11/4): AVX2 with AVX-VNNI where the CPU has it, never AVX-512
    if (v == "avx2e") return 4;     // QuantLab (11/4): the VNNI rows' EVEX test copy (AVX-512 CPUs)
    return 2;
}

template <int KB>
void rows_scalar_t(const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const tq_detail::TqAct* const* a, int nt,
                   float* const* out, int r0, int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        float acc[kTqMaxTokens][4] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const float dx = tq_detail::tq_half_to_float(tq_detail::tq_ld16(blk));
            uint32_t st[32];
            tq_detail::tq_states<KB>(blk + 2, st);
            for (int t = 0; t < nt; ++t) {
                const int16_t* q = a[t]->q + 128 * i;
                for (int k = 0; k < 4; ++k) {
                    int32_t s = 0;
                    for (int j = 0; j < 8; ++j) {
                        const int16_t* wv = tab + 4 * st[8 * k + j];
                        const int16_t* qv = q + 32 * k + 4 * j;
                        s += wv[0] * qv[0] + wv[1] * qv[1] + wv[2] * qv[2] + wv[3] * qv[3];
                    }
                    const float sc = dx * a[t]->d[4 * i + k];
                    acc[t][k] = std::fma((float) s, sc, acc[t][k]);
                }
            }
        }
        for (int t = 0; t < nt; ++t)
            out[t][r] = ((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3])) * (1.0f / 8192.0f);
    }
}

// QuantLab (10/31): rows_scalar_t for R1Q2 blocks (4K code bytes, the scale from tq_detail::TqScales)
template <int KB>
void rows_scalar_r1_t(const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const tq_detail::TqAct* const* a, int nt,
                      float* const* out, int r0, int r1, const tq_detail::TqScales& sc) {
    const int nb = n / 128;
    constexpr size_t bsz = 4 * KB;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        float acc[kTqMaxTokens][4] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const float dx = tq_detail::tq_half_to_float(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3];
            uint32_t st[32];
            tq_detail::tq_states<KB>(blk, st);
            for (int t = 0; t < nt; ++t) {
                const int16_t* q = a[t]->q + 128 * i;
                for (int k = 0; k < 4; ++k) {
                    int32_t s = 0;
                    for (int j = 0; j < 8; ++j) {
                        const int16_t* wv = tab + 4 * st[8 * k + j];
                        const int16_t* qv = q + 32 * k + 4 * j;
                        s += wv[0] * qv[0] + wv[1] * qv[1] + wv[2] * qv[2] + wv[3] * qv[3];
                    }
                    const float scl = dx * a[t]->d[4 * i + k];
                    acc[t][k] = std::fma((float) s, scl, acc[t][k]);
                }
            }
        }
        const float rb = tq_detail::tq_half_to_float(sc.rowbase[r]);
        for (int t = 0; t < nt; ++t)
            out[t][r] = (((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3])) * (1.0f / 8192.0f)) * rb;
    }
}

// TQP (Gyro-L): 64 pair-trellis steps of 2 weights; a 32-value group is 16 steps.  The table row holds the
// TQ2_T state's two pairs; TQP uses the first pair only (weight 2t+j = lut[2*hyb(2*s)+j]).
template <int KB>
void rows_scalar_tqp_t(const uint8_t* w, size_t row_bytes, int n, const tq_detail::TqAct* const* a, int nt,
                       float* const* out, int r0, int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 8 * KB;
    const int16_t* tab = tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        float acc[kTqMaxTokens][4] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const float dx = tq_detail::tq_half_to_float(tq_detail::tq_ld16(blk));
            uint32_t st[64];
            tq_detail::tq_states_tqp<KB>(blk + 2, st);
            for (int t = 0; t < nt; ++t) {
                const int16_t* q = a[t]->q + 128 * i;
                for (int k = 0; k < 4; ++k) {
                    int32_t s = 0;
                    for (int j = 0; j < 16; ++j) {   // step 16k+j: two weights
                        const int16_t* wv = tab + 4 * st[16 * k + j];
                        const int16_t* qv = q + 32 * k + 2 * j;
                        s += wv[0] * qv[0] + wv[1] * qv[1];
                    }
                    const float sc = dx * a[t]->d[4 * i + k];
                    acc[t][k] = std::fma((float) s, sc, acc[t][k]);
                }
            }
        }
        for (int t = 0; t < nt; ++t)
            out[t][r] = ((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3])) * (1.0f / 8192.0f);
    }
}

}  // namespace

const int16_t* tq_state_table() {
    // QuantLab (11/4): refilled when tq_lut_load changed the codebook (at start, before any kernel runs)
    static std::mutex m;
    static std::atomic<uint64_t> filled{~0ull};
    const uint64_t g = strata::kernels::tq_lut_generation();
    if (filled.load(std::memory_order_acquire) != g) {
        std::lock_guard<std::mutex> lk(m);
        if (filled.load(std::memory_order_relaxed) != g) {
            fill_table();
            filled.store(g, std::memory_order_release);
        }
    }
    return g_table;
}

const int16_t* tq_state_table_computed() {
    static std::once_flag once;
    std::call_once(once, fill_table_computed);
    return g_table_c;
}

const int16_t* tq_state_table_bih3() {
    static std::once_flag once;
    std::call_once(once, fill_table_bih3);
    return g_table_b;
}

const int16_t* tq_state_table_for(int t) {
    if (tq_bih3_type(t)) return tq_state_table_bih3();   // QuantLab (11/12)
    return (t == kTqTypeK5X || t == kTqTypeK4X) ? tq_state_table_computed() : tq_state_table();
}

bool tq_supported(int t) noexcept {
    return t == 144 || t == 145 || t == 146 || t == 147 || t == 148 || t == 149 ||
           t == kTqTypeK5X || t == kTqTypeK4X || t == kTqTypeK5T || t == kTqTypeK4T ||
           t == 150 || t == 151 || (t >= kTqPseudoBase + 2 && t <= kTqPseudoBase + 14 && tq_kbits(t) != 0) ||   // QuantLab
           tq_bih3_type(t);   // QuantLab (11/12): bih3
}

// QuantLab: 150 TQK5 and 151 TQK4 (the Gyro SPEC ids) and the local-only pseudo types kTqPseudoBase + K for the rates
// the SPEC has no id for (K 2, 3, 9..14; never handed to ggml).  Upstream's 147..149 (TQP5/6/7) and 224..227 are kept.
int tq_kbits(int t) noexcept {
    if (t == 148) return 6;                                   // TQP6
    if (t == 144) return 8;
    if (t == 146 || t == 149) return 7;
    if (t == 145) return 6;
    if (t == 147 || t == kTqTypeK5X || t == kTqTypeK5T || t == 150) return 5;
    if (t == kTqTypeK4X || t == kTqTypeK4T || t == 151) return 4;
    if (tq_bih3_type(t)) return t - kTqBih3Base;   // QuantLab (11/12): bih3, K 2..14
    const int k = t - kTqPseudoBase;
    if ((k >= 2 && k <= 3) || (k >= 9 && k <= 14)) return k;   // QuantLab, 10/26: K 13, 14 (v2 packs' down rows)
    return 0;
}

bool tq_isa_ok(TqIsa isa) {
#if defined(STRATA_TQ_X86)
    switch (isa) {
        case TqIsa::avx512: return cpu_avx512_ok();
        case TqIsa::avx2: return cpu_avx2_ok();
        case TqIsa::avx2vnni: return cpu_avx2_ok() && tq_detail::tq_avxvnni_built() && cpu_avxvnni_ok();
        case TqIsa::avx2e: return cpu_avx512_ok();   // AVX512-VL + VNNI are part of what cpu_avx512_ok checks
        default: return true;
    }
#else
    return isa == TqIsa::scalar;   // arm64 (the Metal build): the AVX units are not compiled, the scalar kernel runs
#endif
}

TqIsa tq_isa() {
    static const TqIsa isa = [] {
        const int cap = env_isa_cap();
        if (cap == 4) return tq_isa_ok(TqIsa::avx2e) ? TqIsa::avx2e : tq_isa_ok(TqIsa::avx2) ? TqIsa::avx2 : TqIsa::scalar;
        if (cap == 2 && tq_isa_ok(TqIsa::avx512)) return TqIsa::avx512;
        if ((cap == 2 || cap == 3) && tq_isa_ok(TqIsa::avx2vnni)) return TqIsa::avx2vnni;   // QuantLab (11/4)
        if (cap >= 1 && tq_isa_ok(TqIsa::avx2)) return TqIsa::avx2;
        return TqIsa::scalar;
    }();
    return isa;
}

bool tq_kernels_enabled() {
    static const bool on = [] {
        if (const char* e = std::getenv("STRATA_NO_TQ_KERNELS"); e != nullptr && std::atoi(e) != 0) return false;
        // Every CPU runs them: the scalar kernel (baseline x86-64) where there is no AVX2.  A ggml built from upstream
        // has no vec_dot for the trellis types to fall back to, so a CPU without AVX2 must not be sent there.
        return true;
    }();
    return on;
}

const char* tq_isa_name(TqIsa isa) {
    return isa == TqIsa::avx512 ? "avx512" : isa == TqIsa::avx2 ? "avx2" : isa == TqIsa::avx2vnni ? "avxvnni"
           : isa == TqIsa::avx2e ? "avx2e" : "scalar";
}

namespace tq_detail {

void unpack_act(const void* q8_0, int n, TqAct& a) {
    const uint8_t* p = (const uint8_t*) q8_0;
    for (int b = 0; b < n / 32; ++b, p += 34) {   // block_q8_0: fp16 d, 32 x int8
        a.d[b] = tq_half_to_float(tq_ld16(p));
        const int8_t* q = (const int8_t*) (p + 2);
        int32_t sum = 0;
        for (int j = 0; j < 32; ++j) {
            a.q[32 * b + j] = q[j];
            a.q8[32 * b + j] = q[j];   // QuantLab (11/12): the bih3 rows' side
            sum += q[j];
        }
        a.bsum[b] = 384 * sum;
        for (int g = 0; g < 8; ++g) {
            a.c4[8 * b + g] = -384 * ((int32_t) q[4 * g] + q[4 * g + 1] + q[4 * g + 2] + q[4 * g + 3]);
            a.d4[8 * b + g] = a.d[b];
            const int j = 8 * b + g;   // the state's index in the row
            a.qp0[2 * j] = q[4 * g];
            a.qp0[2 * j + 1] = q[4 * g + 1];
            a.qp1[2 * j] = q[4 * g + 2];
            a.qp1[2 * j + 1] = q[4 * g + 3];
        }
    }
}

void tq_rows_scalar(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1) {
    STRATA_TQ_DISPATCH_K(kb, rows_scalar_t, tab, w, row_bytes, n, a, nt, out, r0, r1)
}

void tq_rows_scalar_r1q2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                         int nt, float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, rows_scalar_r1_t, tab, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_scalar_tqp(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                        float* const* out, int r0, int r1) {
    if (kb == 5) rows_scalar_tqp_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_scalar_tqp_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_scalar_tqp_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace tq_detail

namespace {

// the activations of one call, unpacked once for all its rows
struct ActSet {
    tq_detail::TqAct act[kTqMaxTokens];
    const tq_detail::TqAct* ptr[kTqMaxTokens];
};

// QuantLab (11/13) STRATA_QL_UNPACK_ONCE=1: the pool's tasks each unpacked their tokens' activations again (every task,
// every expert of its row range).  With it a thread keeps what it unpacked, keyed by the source pointer and the batch's
// generation (tq_set_act_generation; the pool sets it per phase, 0 = no reuse), and the AVX-512 bih3 exact rows - which
// read only q8, d and bsum - get only those fields filled.  The same values: the rows' bits do not change.
bool unpack_once_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_QL_UNPACK_ONCE"); return v && v[0] == '1'; }();
    return on;
}
thread_local uint64_t tl_act_gen = 0;

struct ActSlot {
    const void* src = nullptr;
    uint64_t gen = 0;
    int n = 0;
    bool lean = false;
    std::unique_ptr<tq_detail::TqAct> act;
};
constexpr int kActSlots = 16;

void unpack_act_lean(const void* q8_0, int n, tq_detail::TqAct& a) {   // unpack_act's d, q8 and bsum only
    const uint8_t* p = (const uint8_t*) q8_0;
    for (int b = 0; b < n / 32; ++b, p += 34) {
        a.d[b] = tq_detail::tq_half_to_float(tq_detail::tq_ld16(p));
        const int8_t* q = (const int8_t*) (p + 2);
        int32_t sum = 0;
        for (int j = 0; j < 32; ++j) {
            a.q8[32 * b + j] = q[j];
            sum += q[j];
        }
        a.bsum[b] = 384 * sum;
    }
}

ActSet& unpack_all(const void* const* act, int n, int nt, bool lean = false) {
    thread_local std::unique_ptr<ActSet> set;
    if (!set) set.reset(new ActSet);
    if (!unpack_once_on()) {
        for (int t = 0; t < nt; ++t) {
            tq_detail::unpack_act(act[t], n, set->act[t]);
            set->ptr[t] = &set->act[t];
        }
        return *set;
    }
    thread_local ActSlot slots[kActSlots];
    thread_local int next = 0;
    const uint64_t gen = tl_act_gen;
    for (int t = 0; t < nt; ++t) {
        ActSlot* hit = nullptr;
        if (gen != 0)
            for (ActSlot& s : slots)
                if (s.act && s.gen == gen && s.src == act[t] && s.n == n && (lean || !s.lean)) { hit = &s; break; }
        if (hit == nullptr) {
            // a slot not used by this call's earlier tokens (nt <= 8 < kActSlots)
            for (;;) {
                ActSlot& s = slots[next];
                next = (next + 1) % kActSlots;
                bool taken = false;
                for (int u = 0; u < t; ++u) taken = taken || set->ptr[u] == s.act.get();
                if (taken) continue;
                hit = &s;
                break;
            }
            if (!hit->act) hit->act.reset(new tq_detail::TqAct);
            if (lean) unpack_act_lean(act[t], n, *hit->act);
            else tq_detail::unpack_act(act[t], n, *hit->act);
            hit->src = act[t];
            hit->gen = gen;
            hit->n = n;
            hit->lean = lean;
        }
        set->ptr[t] = hit->act.get();
    }
    return *set;
}

void rows_on(TqIsa isa, int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n,
             const tq_detail::TqAct* const* a, int nt, float* const* out, int r0, int r1) {
    switch (isa) {
#if defined(STRATA_TQ_X86)
        case TqIsa::avx512: tq_detail::tq_rows_avx512(kb, tab, w, row_bytes, n, a, nt, out, r0, r1); break;
        case TqIsa::avx2: tq_detail::tq_rows_avx2(kb, tab, w, row_bytes, n, a, nt, out, r0, r1); break;
        case TqIsa::avx2vnni: tq_detail::tq_rows_avxvnni(kb, tab, w, row_bytes, n, a, nt, out, r0, r1); break;
        case TqIsa::avx2e: tq_detail::tq_rows_avx2e(kb, tab, w, row_bytes, n, a, nt, out, r0, r1); break;
#endif
        default: tq_detail::tq_rows_scalar(kb, tab, w, row_bytes, n, a, nt, out, r0, r1); break;
    }
}

// The pair layout (types 147-149): AVX2 and AVX-512 both take the AVX2 pair kernel; scalar, and any build without the
// x86 units (arm64: the Metal build), the scalar one.
void rows_pair_on(TqIsa isa, int kb, const uint8_t* w, size_t row_bytes, int n, const tq_detail::TqAct* const* a, int nt,
                  float* const* out, int r0, int r1) {
#if defined(STRATA_TQ_X86)
    if (isa != TqIsa::scalar) {
        tq_detail::tq_rows_avx2_tqp(kb, w, row_bytes, n, a, nt, out, r0, r1);
        return;
    }
#else
    (void) isa;
#endif
    tq_detail::tq_rows_scalar_tqp(kb, w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace

void tq_rows(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1) {
    if (!tq_isa_ok(isa)) std::abort();   // a caller bug: asked for an ISA this CPU does not have
    const int16_t* tab = tq_state_table_for(ggml_type);
    const ActSet& s = unpack_all(act, n, nt);
    if (ggml_type >= 147 && ggml_type <= 149) {   // Gyro-L pair trellis: the AVX2 pair kernel (scalar is the reference)
        const int kb = tq_kbits(ggml_type);
        rows_pair_on(isa, kb, w, row_bytes, n, s.ptr, nt, out, r0, r1);
        return;
    }
    rows_on(isa, tq_kbits(ggml_type), tab, w, row_bytes, n, s.ptr, nt, out, r0, r1);
}

// QuantLab (11/12): STRATA_TQ_BIH3_TABLE=1 - the bih3 types on the table rows on AVX-512 too (A/B; the same bits)
static bool bih3_table_only() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TQ_BIH3_TABLE"); return v && v[0] == '1'; }();
    return on;
}
static bool bih3_lanes() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TQ_BIH3_LANES"); return v && v[0] == '1'; }();
    return on;
}
// QuantLab (11/13) STRATA_QL_UNPACK_ONCE: the rows rows_r1_on runs read only q8, d and bsum (the AVX-512 bih3 exact rows)
static bool lean_act(TqIsa isa, int ggml_type) {
    return unpack_once_on() && isa == TqIsa::avx512 && tq_bih3_type(ggml_type) && !bih3_table_only() && !bih3_lanes();
}
// QuantLab (10/31): R1Q2 blocks (4K bytes, tq_detail::TqScales), each ISA its own kernel as rows_on
void rows_r1_on(TqIsa isa, int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n,
                const tq_detail::TqAct* const* a, int nt, float* const* out, int r0, int r1, const tq_detail::TqScales& sc,
                bool bih3 = false) {
#if defined(STRATA_TQ_X86)
    if (isa == TqIsa::avx512 && bih3 && !bih3_table_only()) {   // QuantLab (11/12): computed, no table
        if (bih3_lanes()) tq_detail::tq_rows_avx512_r1q2_bih3_lanes(kb, w, row_bytes, n, a, nt, out, r0, r1, sc);
        else tq_detail::tq_rows_avx512_r1q2_bih3(kb, w, row_bytes, n, a, nt, out, r0, r1, sc);
        return;
    }
    // STRATA_TQ_BIH3_AVX2: 1 = every bih3 row computed, 2 = only the wide rows (n >= 2048: gate / up, two thirds of
    // the work; the 640-wide down rows keep the table: on Zen 4 the computed rows were 9-18% faster at n 2560 and
    // 0-13% slower at n 640, strata-desktop's split)
    static const int avx2_compute = [] { const char* v = std::getenv("STRATA_TQ_BIH3_AVX2"); return v ? std::atoi(v) : 0; }();
    if (bih3 && (avx2_compute == 1 || (avx2_compute == 2 && n >= 2048)) && !bih3_table_only() &&
        (isa == TqIsa::avx2 || isa == TqIsa::avx2vnni || isa == TqIsa::avx2e)) {
        // QuantLab (11/12): the AVX2 bih3 rows (and their AVX-VNNI / EVEX copies), opt-in: on Zen 4 they measured
        // the table rows' speed (-17 .. +15%), so the table rows stay the default until a Raptor Lake measures them
        if (isa == TqIsa::avx2vnni) tq_detail::tq_rows_avxvnni_r1q2_bih3(kb, w, row_bytes, n, a, nt, out, r0, r1, sc);
        else if (isa == TqIsa::avx2e) tq_detail::tq_rows_avx2e_r1q2_bih3(kb, w, row_bytes, n, a, nt, out, r0, r1, sc);
        else tq_detail::tq_rows_avx2_r1q2_bih3(kb, w, row_bytes, n, a, nt, out, r0, r1, sc);
        return;
    }
    if (isa == TqIsa::avx512) {
        tq_detail::tq_rows_avx512_r1q2(kb, tab, w, row_bytes, n, a, nt, out, r0, r1, sc);
        return;
    }
    if (isa == TqIsa::avx2) {
        tq_detail::tq_rows_avx2_r1q2(kb, tab, w, row_bytes, n, a, nt, out, r0, r1, sc);
        return;
    }
    if (isa == TqIsa::avx2vnni) {   // QuantLab (11/4)
        tq_detail::tq_rows_avxvnni_r1q2(kb, tab, w, row_bytes, n, a, nt, out, r0, r1, sc);
        return;
    }
    if (isa == TqIsa::avx2e) {
        tq_detail::tq_rows_avx2e_r1q2(kb, tab, w, row_bytes, n, a, nt, out, r0, r1, sc);
        return;
    }
#endif
    (void) isa;
    tq_detail::tq_rows_scalar_r1q2(kb, tab, w, row_bytes, n, a, nt, out, r0, r1, sc);
}
void tq_rows_r1q2(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1, const tq_detail::TqScales& sc) {
    if (!tq_isa_ok(isa) || (ggml_type >= 147 && ggml_type <= 149)) std::abort();   // a caller bug
    const ActSet& s = unpack_all(act, n, nt, lean_act(isa, ggml_type));
    rows_r1_on(isa, tq_kbits(ggml_type), tq_state_table_for(ggml_type), w, row_bytes, n, s.ptr, nt, out, r0, r1, sc,
               tq_bih3_type(ggml_type));
}
void tq_gu_rows_r1q2(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1, const tq_detail::TqScales& g,
                     const tq_detail::TqScales& u) {
    if (!tq_isa_ok(isa) || (ggml_type >= 147 && ggml_type <= 149)) std::abort();
    const int16_t* tab = tq_state_table_for(ggml_type);
    const ActSet& s = unpack_all(act, n, nt, lean_act(isa, ggml_type));
    thread_local std::vector<float> gbuf, ubuf;
    const size_t nr = (size_t) r1;
    if (gbuf.size() < nr * kTqMaxTokens) { gbuf.resize(nr * kTqMaxTokens); ubuf.resize(nr * kTqMaxTokens); }
    float* gp[kTqMaxTokens];
    float* up[kTqMaxTokens];
    for (int t = 0; t < nt; ++t) {
        gp[t] = gbuf.data() + (size_t) t * nr;
        up[t] = ubuf.data() + (size_t) t * nr;
    }
    const int kb = tq_kbits(ggml_type);
    rows_r1_on(isa, kb, tab, blob, gu_row, n, s.ptr, nt, gp, r0, r1, g, tq_bih3_type(ggml_type));
    rows_r1_on(isa, kb, tab, blob + up_off, gu_row, n, s.ptr, nt, up, r0, r1, u, tq_bih3_type(ggml_type));
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float gv = gp[t][r], uv = up[t][r];
            ff[t][r] = (gv / (1.f + std::exp(-gv))) * uv;   // tq_gu_rows' SwiGLU expression
        }
}

void tq_gu_rows(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                const void* const* act, int nt, float* const* ff, int r0, int r1) {
    if (!tq_isa_ok(isa)) std::abort();
    const int16_t* tab = tq_state_table_for(ggml_type);
    const ActSet& s = unpack_all(act, n, nt);
    thread_local std::vector<float> gbuf, ubuf;
    const size_t nr = (size_t) r1;   // row r of token t at [t * r1 + r]
    if (gbuf.size() < nr * kTqMaxTokens) { gbuf.resize(nr * kTqMaxTokens); ubuf.resize(nr * kTqMaxTokens); }
    float* gp[kTqMaxTokens];
    float* up[kTqMaxTokens];
    for (int t = 0; t < nt; ++t) {
        gp[t] = gbuf.data() + (size_t) t * nr;
        up[t] = ubuf.data() + (size_t) t * nr;
    }
    const int kb = tq_kbits(ggml_type);
    const bool pair = ggml_type >= 147 && ggml_type <= 149;   // Gyro-L pair trellis: its own kernels
    auto rows = [&](const uint8_t* w, float* const* o) {
        if (pair) rows_pair_on(isa, kb, w, gu_row, n, s.ptr, nt, o, r0, r1);
        else rows_on(isa, kb, tab, w, gu_row, n, s.ptr, nt, o, r0, r1);
    };
    rows(blob, gp);
    rows(blob + up_off, up);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float g = gp[t][r], u = up[t][r];
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;   // native_gu_rows' SwiGLU expression
        }
}

void tq_set_act_generation(uint64_t gen) { tl_act_gen = gen; }

}  // namespace strata::kernels::cpu
