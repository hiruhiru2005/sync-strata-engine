// src/kernels/cpu/tq_avx2.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7) on AVX2 + FMA (tq_cpu.hpp).
// Compiled for AVX2 only and reached behind cpu_avx2_ok(), so nothing here can fault on an older CPU.
//
// Per 128-weight block: the 32 states in scalar code (tq_states.hpp: one 32-bit read, a shift and a mask each, used
// as table indices straight away), four 8-byte state-table rows per YMM register as scalar loads, then per token
// VPMADDWD against the unpacked int16 activations and a VPHADDD tree that leaves the four exact sub-block sums
// S[k] in one XMM, so the float part is the scalar kernel's, lane for lane.  The weights are decoded once per block
// for all the tokens of the group.
//
// Measured on Zen 5 (cycles per block, one token, AVX2): ~48 with this scheme; computing the states with byte
// shuffles instead (VPSHUFB / VPSRLVD per 8 states) and moving them to GPRs or through memory was slower (60-80),
// and ggml-cpu's AVX2 vec_dot for these types (agentionai llama.cpp, the same state table) takes about 2.3x as long.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

namespace strata::kernels::cpu::tq_detail {
namespace {

// STRATA_AVXVNNI (CMake: the compiler has the AVX-VNNI intrinsics), as in iq_avx2.cpp: GCC and Clang compile the VNNI
// copy only inside functions with target("avxvnni"); MSVC takes the intrinsics under /arch:AVX2.
#if !defined(STRATA_AVXVNNI)
#define STRATA_AVXVNNI 0
#endif
#if STRATA_AVXVNNI && (defined(__GNUC__) || defined(__clang__))
#define STRATA_AVXVNNI_FN __attribute__((target("avxvnni")))
#else
#define STRATA_AVXVNNI_FN
#endif

// ---- the TQ2_T / TQK and R1Q2 rows, twice: plain AVX2 and the AVX-VNNI copy (tq_avx2_rows.inl)
namespace plain {
#define STRATA_TQ_VNNI 0
#define STRATA_TQ_ROWS_FN
#include "tq_avx2_rows.inl"
}  // namespace plain
#if STRATA_AVXVNNI
namespace vnni {
#define STRATA_TQ_VNNI 1
#define STRATA_TQ_ROWS_FN STRATA_AVXVNNI_FN
#include "tq_avx2_rows.inl"
}  // namespace vnni
#endif

// TQP (147-149): 64 pair-steps per 128-value block.  A step's two weights are the FIRST pair of its state
// table row, so 8 states make one YMM of 16 int16 weights, and a 32-value group (16 steps) is two YMM of
// weights against two YMM of activations: VPMADDWD, one exact int32 sum per group, then the four group sums
// go into the same per-token __m128 the other kernels use, so the float part is the scalar kernel's.
inline __m256i pairs8(const int16_t* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, uint32_t s4,
                      uint32_t s5, uint32_t s6, uint32_t s7) {
    const __m128i a0 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s0)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s1)));
    const __m128i a1 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s2)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s3)));
    const __m128i a2 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s4)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s5)));
    const __m128i a3 = _mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*) (tab + 4 * s6)),
                                          _mm_loadl_epi64((const __m128i*) (tab + 4 * s7)));
    return _mm256_set_m128i(_mm_unpacklo_epi64(a2, a3), _mm_unpacklo_epi64(a0, a1));
}

template <int KB>
void rows_tqp_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
                int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 8 * KB;
    const int16_t* tab = tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            uint32_t st[64];
            tq_states_tqp<KB>(blk + 2, st);
            __m256i wv[8];   // wv[2g], wv[2g+1]: the weights of the 32-value group g
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                const uint32_t* s = st + 16 * g;
                wv[2 * g] = pairs8(tab, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
                wv[2 * g + 1] = pairs8(tab, s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]);
            }
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m256i* q = (const __m256i*) (a[t]->q + 128 * i);
                int32_t sum[4];
#pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wv[2 * g], _mm256_load_si256(q + 2 * g)),
                                                       _mm256_madd_epi16(wv[2 * g + 1], _mm256_load_si256(q + 2 * g + 1)));
                    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
                    s = _mm_hadd_epi32(s, s);
                    s = _mm_hadd_epi32(s, s);
                    sum[g] = _mm_cvtsi128_si32(s);
                }
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                const __m128i s4 = _mm_set_epi32(sum[3], sum[2], sum[1], sum[0]);
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s4), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

}  // namespace

void tq_rows_avx2_tqp(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                      float* const* out, int r0, int r1) {
    if (kb == 5) rows_tqp_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_tqp_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_tqp_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

void tq_rows_avx2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1) {
    STRATA_TQ_DISPATCH_K(kb, plain::rows_t, table, w, row_bytes, n, a, nt, out, r0, r1)
}
void tq_rows_avx2_r1q2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                       float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, plain::rows_r1_t, table, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_avx2_r1q2_bih3(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                            float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, plain::rows_r1_bih3_t, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_avxvnni_r1q2_bih3(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                               float* const* out, int r0, int r1, const TqScales& sc) {
#if STRATA_AVXVNNI
    STRATA_TQ_DISPATCH_K(kb, vnni::rows_r1_bih3_t, w, row_bytes, n, a, nt, out, r0, r1, sc)
#else
    tq_rows_avx2_r1q2_bih3(kb, w, row_bytes, n, a, nt, out, r0, r1, sc);
#endif
}

// QuantLab (11/4): the AVX-VNNI copies (callers check tq_isa_ok(TqIsa::avx2vnni) first; a build without the
// intrinsics runs the plain rows here)
bool tq_avxvnni_built() { return STRATA_AVXVNNI != 0; }
void tq_rows_avxvnni(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                     float* const* out, int r0, int r1) {
#if STRATA_AVXVNNI
    STRATA_TQ_DISPATCH_K(kb, vnni::rows_t, table, w, row_bytes, n, a, nt, out, r0, r1)
#else
    tq_rows_avx2(kb, table, w, row_bytes, n, a, nt, out, r0, r1);
#endif
}
void tq_rows_avxvnni_r1q2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                          int nt, float* const* out, int r0, int r1, const TqScales& sc) {
#if STRATA_AVXVNNI
    STRATA_TQ_DISPATCH_K(kb, vnni::rows_r1_t, table, w, row_bytes, n, a, nt, out, r0, r1, sc)
#else
    tq_rows_avx2_r1q2(kb, table, w, row_bytes, n, a, nt, out, r0, r1, sc);
#endif
}

}  // namespace strata::kernels::cpu::tq_detail
