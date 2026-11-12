// src/kernels/cpu/tq_avx512.cpp - the trellis expert rows (TQ2_T / TQK6 / TQK7) on AVX-512 F/BW/VL/VNNI
// (tq_cpu.hpp).  Reached behind cpu_avx512_ok() only.
//
// Per 128-weight block: the AVX2 kernel's decode (scalar states, scalar 8-byte state-table loads, one ZMM of 32
// int16 weights per Q8_0 sub-block); per token four VPDPWSSD against the unpacked int16 activations, an unpack/add
// tree to the four exact sub-block sums S[k] in one XMM, and the scalar kernel's float steps lane for lane.  The
// weights are decoded once per block for all the tokens of the group.
//
// Measured on Zen 5 (cycles per block): the decode bounds it, so one token is no faster than AVX2 (~48); each extra
// token of a group costs ~10% less than on AVX2.  ggml-cpu's AVX-512 decode for these types - all states from two
// VPERMB and the table rows by VPGATHERQQ - was 20-30% slower here: Zen gathers are slow, and so are Intel ones
// under the Gather Data Sampling microcode mitigation.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

namespace strata::kernels::cpu::tq_detail {
namespace {

// four ZMM of 16 int32 partial sums -> [S0, S1, S2, S3] (exact integer adds)
inline __m128i reduce4(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i a = _mm512_add_epi32(_mm512_unpacklo_epi32(p0, p1), _mm512_unpackhi_epi32(p0, p1));
    const __m512i b = _mm512_add_epi32(_mm512_unpacklo_epi32(p2, p3), _mm512_unpackhi_epi32(p2, p3));
    const __m512i c = _mm512_add_epi32(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
    // c: per 128-bit lane [p0, p1, p2, p3] partial sums; add the four lanes
    const __m256i d = _mm256_add_epi32(_mm512_castsi512_si256(c), _mm512_extracti64x4_epi64(c, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
}

// four state-table rows (8 bytes each) into one YMM: scalar loads, broadcast and blended in; two make a ZMM
inline __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

template <int KB>
inline __m512i rows8(const long long* tab, const uint8_t* qs, uint64_t wr, int t0) {
    const __m256i lo = rows4(tab, tq_state<KB>(qs, wr, t0), tq_state<KB>(qs, wr, t0 + 1), tq_state<KB>(qs, wr, t0 + 2),
                             tq_state<KB>(qs, wr, t0 + 3));
    const __m256i hi = rows4(tab, tq_state<KB>(qs, wr, t0 + 4), tq_state<KB>(qs, wr, t0 + 5),
                             tq_state<KB>(qs, wr, t0 + 6), tq_state<KB>(qs, wr, t0 + 7));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

template <int KB>
void rows_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
            int r1) {
    const int nb = n / 128;
    constexpr size_t bsz = 2 + 4 * KB;
    const long long* tab = (const long long*) table;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const uint8_t* qs = blk + 2;
            const uint64_t wr = tq_wrap<KB>(qs);
            // states 8k .. 8k+7 -> weights 32k .. 32k+31 = Q8_0 sub-block k
            const __m512i w0 = rows8<KB>(tab, qs, wr, 0), w1 = rows8<KB>(tab, qs, wr, 8);
            const __m512i w2 = rows8<KB>(tab, qs, wr, 16), w3 = rows8<KB>(tab, qs, wr, 24);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const __m512i* q = (const __m512i*) (a[t]->q + 128 * i);
                const __m512i z = _mm512_setzero_si512();
                const __m128i s = reduce4(_mm512_dpwssd_epi32(z, w0, _mm512_load_si512(q + 0)),
                                          _mm512_dpwssd_epi32(z, w1, _mm512_load_si512(q + 1)),
                                          _mm512_dpwssd_epi32(z, w2, _mm512_load_si512(q + 2)),
                                          _mm512_dpwssd_epi32(z, w3, _mm512_load_si512(q + 3)));
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

// QuantLab (10/31): rows_t for R1Q2 blocks (4K code bytes; the block scale colscale[b] * p2[q], rowbase at the end)
template <int KB>
void rows_r1_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
               float* const* out, int r0, int r1, const TqScales& sc) {
    const int nb = n / 128;
    constexpr size_t bsz = 4 * KB;
    const long long* tab = (const long long*) table;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* qs = x + (size_t) i * bsz;
            const uint64_t wr = tq_wrap<KB>(qs);
            const __m512i w0 = rows8<KB>(tab, qs, wr, 0), w1 = rows8<KB>(tab, qs, wr, 8);
            const __m512i w2 = rows8<KB>(tab, qs, wr, 16), w3 = rows8<KB>(tab, qs, wr, 24);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3]);
            for (int t = 0; t < nt; ++t) {
                const __m512i* q = (const __m512i*) (a[t]->q + 128 * i);
                const __m512i z = _mm512_setzero_si512();
                const __m128i s = reduce4(_mm512_dpwssd_epi32(z, w0, _mm512_load_si512(q + 0)),
                                          _mm512_dpwssd_epi32(z, w1, _mm512_load_si512(q + 1)),
                                          _mm512_dpwssd_epi32(z, w2, _mm512_load_si512(q + 2)),
                                          _mm512_dpwssd_epi32(z, w3, _mm512_load_si512(q + 3)));
                const __m128 scl = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), scl, acc[t]);
            }
        }
        const float rb = tq_half_to_float(sc.rowbase[r]);
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = (((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f)) * rb;
        }
    }
}

// QuantLab (11/12): the bih3 R1Q2 rows (types 180 + K), the points computed instead of looked up (bench_decode's SM 6,
// measured ~2x the table rows on Zen 5 for K 8).  Per block: the 32 states (scalar, as rows8), 16 a ZMM; x_i = s * C_i
// (VPMULLD); the planes T0 = [x0.b1 x0.b2 x0.b3 x1.b1], T1 = [x1.b2 x1.b3 x2.b1 x2.b2], T2 = [x2.b3 x3.b1 x3.b2 x3.b3]
// (unsigned bytes) by VPTERNLOG; per token VPDPBUSD of each plane against the int8 activations, the lanes summed per
// Q8_0 sub-block, minus 384 * the sub-block's sum: S'[k] = sum (b0 + b1 + b2 - 384) * q, exact.  The table rows' S[k]
// is 64 * S'[k] exactly (their weights are (sum - 384) * 64), so after the shift the float steps are the scalar
// kernel's lane for lane: bitwise equal.
// the per-sub-block sums of A (states 0..15) and B (16..31): [A lanes 0-7, A 8-15, B 0-7, B 8-15]
inline __m128i reduce_halves(__m512i A, __m512i B) {
    const __m256i x = _mm256_hadd_epi32(_mm512_castsi512_si256(A), _mm512_extracti64x4_epi64(A, 1));
    const __m256i y = _mm256_hadd_epi32(_mm512_castsi512_si256(B), _mm512_extracti64x4_epi64(B, 1));
    const __m256i z = _mm256_hadd_epi32(x, y);
    return _mm_add_epi32(_mm256_castsi256_si128(z), _mm256_extracti128_si256(z, 1));
}
// The 32 states of a block in two ZMM without a trip through memory: the block's 4K bytes in one ZMM (masked load),
// per lane t one VPERMB of the 4 bytes from byte (31 - t) K / 8 on (circular over the 4K bytes, as the stream is), a
// VPSRLVD by ((31 - t) K) & 7, the 15-bit mask - tq_state's bits.  K 8 (TQ2_T's byte states): bytes t and t - 1.
template <int KB>
struct Bih3States {
    alignas(64) uint8_t idx[2][64];
    alignas(64) uint32_t sh[2][16];
    Bih3States() {
        constexpr int NB = 4 * KB;
        for (int g = 0; g < 2; ++g)
            for (int l = 0; l < 16; ++l) {
                const int t = 16 * g + l;
                if (KB == 8) {
                    idx[g][4 * l + 0] = (uint8_t) t;
                    idx[g][4 * l + 1] = (uint8_t) ((t + 31) & 31);
                    idx[g][4 * l + 2] = idx[g][4 * l + 3] = 0;
                    sh[g][l] = 0;
                } else {
                    const int off = (31 - t) * KB, b = off >> 3;
                    for (int j = 0; j < 4; ++j) idx[g][4 * l + j] = (uint8_t) ((b + j) % NB);
                    sh[g][l] = (uint32_t) (off & 7);
                }
            }
    }
};
template <int KB>
void rows_r1_bih3_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
                    int r1, const TqScales& sc) {
    const int nb = n / 128;
    constexpr size_t bsz = 4 * KB;
    static const Bih3States<KB> S;
    const __m512i ix0 = _mm512_load_si512(S.idx[0]), ix1 = _mm512_load_si512(S.idx[1]);
    const __m512i sh0 = _mm512_load_si512(S.sh[0]), sh1 = _mm512_load_si512(S.sh[1]);
    const __m512i m15 = _mm512_set1_epi32(0x7FFF);
    const __mmask64 lm = (__mmask64) (bsz >= 64 ? ~0ull : (1ull << bsz) - 1);
    const __m512i c0 = _mm512_set1_epi32((int) 0x9E3779B1u), c1 = _mm512_set1_epi32((int) 0x85EBCA6Bu);
    const __m512i c2 = _mm512_set1_epi32((int) 0xC2B2AE35u), c3 = _mm512_set1_epi32((int) 0x27D4EB2Fu);
    const __m512i m24 = _mm512_set1_epi32((int) 0xFF000000u), m16 = _mm512_set1_epi32((int) 0xFFFF0000u);
    const __m512i m8 = _mm512_set1_epi32((int) 0xFFFFFF00u);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* qs = x + (size_t) i * bsz;
            const __m512i blk = _mm512_maskz_loadu_epi8(lm, qs);
            __m512i u[6];
            for (int g = 0; g < 2; ++g) {
                const __m512i s = _mm512_and_si512(
                    _mm512_srlv_epi32(_mm512_permutexvar_epi8(g ? ix1 : ix0, blk), g ? sh1 : sh0), m15);
                const __m512i x0 = _mm512_mullo_epi32(s, c0), x1 = _mm512_mullo_epi32(s, c1);
                const __m512i x2 = _mm512_mullo_epi32(s, c2), x3 = _mm512_mullo_epi32(s, c3);
                // VPTERNLOG 0xEA = (a & b) | c
                u[3 * g] = _mm512_ternarylogic_epi32(m24, _mm512_slli_epi32(x1, 16), _mm512_srli_epi32(x0, 8), 0xEA);
                u[3 * g + 1] = _mm512_ternarylogic_epi32(m16, _mm512_slli_epi32(x2, 8), _mm512_srli_epi32(x1, 16), 0xEA);
                u[3 * g + 2] = _mm512_ternarylogic_epi32(m8, x3, _mm512_srli_epi32(x2, 24), 0xEA);
            }
            const __m128 dx = _mm_set1_ps(tq_half_to_float(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3]);
            for (int t = 0; t < nt; ++t) {
                const int8_t* a8 = a[t]->q8 + 128 * i;
                const __m512i qa = _mm512_load_si512(a8), qb = _mm512_load_si512(a8 + 64);
                const __m512i z = _mm512_setzero_si512();
                __m512i A = _mm512_dpbusd_epi32(z, u[0], qa);
                A = _mm512_dpbusd_epi32(A, u[1], qa);
                A = _mm512_dpbusd_epi32(A, u[2], qa);
                __m512i B = _mm512_dpbusd_epi32(z, u[3], qb);
                B = _mm512_dpbusd_epi32(B, u[4], qb);
                B = _mm512_dpbusd_epi32(B, u[5], qb);
                const __m128i s1 = _mm_sub_epi32(reduce_halves(A, B), _mm_load_si128((const __m128i*) (a[t]->bsum + 4 * i)));
                const __m128i s = _mm_slli_epi32(s1, 6);   // the table rows' S[k] (weights x 2^13)
                const __m128 scl = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), scl, acc[t]);
            }
        }
        const float rb = tq_half_to_float(sc.rowbase[r]);
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = (((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f)) * rb;
        }
    }
}

// QuantLab (11/13) STRATA_QL_TQ_PRE=1: rows_r1_bih3_t with its per-block scalar work out of the loops and two tokens
// folded together.  The block scales colscale[i] * p2[q] are one float product, made once per call (4 per block) instead
// of once per block and row (a software fp16 conversion each); the tokens' pointers are read once; a token pair's two
// reduce_halves share their last fold (VPERM2I128 + one add for both) and their float steps run in one YMM, lane for lane
// the same operations as the XMM ones.  Integer sums and the float order unchanged: bitwise equal to rows_r1_bih3_t.
inline __m256i halves8(__m512i A, __m512i B) {   // reduce_halves before its last fold: S[k] = lane k + lane k + 4
    const __m256i x = _mm256_hadd_epi32(_mm512_castsi512_si256(A), _mm512_extracti64x4_epi64(A, 1));
    const __m256i y = _mm256_hadd_epi32(_mm512_castsi512_si256(B), _mm512_extracti64x4_epi64(B, 1));
    return _mm256_hadd_epi32(x, y);
}
template <int KB>
void rows_r1_bih3_pre_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out,
                        int r0, int r1, const TqScales& sc) {
    const int nb = n / 128;
    constexpr size_t bsz = 4 * KB;
    static const Bih3States<KB> S;
    const __m512i ix0 = _mm512_load_si512(S.idx[0]), ix1 = _mm512_load_si512(S.idx[1]);
    const __m512i sh0 = _mm512_load_si512(S.sh[0]), sh1 = _mm512_load_si512(S.sh[1]);
    const __m512i m15 = _mm512_set1_epi32(0x7FFF);
    const __mmask64 lm = (__mmask64) (bsz >= 64 ? ~0ull : (1ull << bsz) - 1);
    const __m512i c0 = _mm512_set1_epi32((int) 0x9E3779B1u), c1 = _mm512_set1_epi32((int) 0x85EBCA6Bu);
    const __m512i c2 = _mm512_set1_epi32((int) 0xC2B2AE35u), c3 = _mm512_set1_epi32((int) 0x27D4EB2Fu);
    const __m512i m24 = _mm512_set1_epi32((int) 0xFF000000u), m16 = _mm512_set1_epi32((int) 0xFFFF0000u);
    const __m512i m8 = _mm512_set1_epi32((int) 0xFFFFFF00u);
    alignas(64) float dxt[(kTqMaxN / 128) * 4];   // block i, code q: colscale[i] * p2[q] (rows_r1_bih3_t's product)
    for (int i = 0; i < nb; ++i) {
        const float cs = tq_half_to_float(sc.colscale[i]);
        for (int q = 0; q < 4; ++q) dxt[4 * i + q] = cs * sc.p2[q];
    }
    const int8_t* q8[kTqMaxTokens];
    const int32_t* bs[kTqMaxTokens];
    const float* dd[kTqMaxTokens];
    for (int t = 0; t < nt; ++t) { q8[t] = a[t]->q8; bs[t] = a[t]->bsum; dd[t] = a[t]->d; }
    const int np = nt / 2;   // token pairs; an odd last token on the XMM steps
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        __m256 acc2[kTqMaxTokens / 2];
        for (int p = 0; p < np; ++p) acc2[p] = _mm256_setzero_ps();
        __m128 acc1 = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const __m512i blk = _mm512_maskz_loadu_epi8(lm, x + (size_t) i * bsz);
            __m512i u[6];
            for (int g = 0; g < 2; ++g) {
                const __m512i s = _mm512_and_si512(
                    _mm512_srlv_epi32(_mm512_permutexvar_epi8(g ? ix1 : ix0, blk), g ? sh1 : sh0), m15);
                const __m512i x0 = _mm512_mullo_epi32(s, c0), x1 = _mm512_mullo_epi32(s, c1);
                const __m512i x2 = _mm512_mullo_epi32(s, c2), x3 = _mm512_mullo_epi32(s, c3);
                u[3 * g] = _mm512_ternarylogic_epi32(m24, _mm512_slli_epi32(x1, 16), _mm512_srli_epi32(x0, 8), 0xEA);
                u[3 * g + 1] = _mm512_ternarylogic_epi32(m16, _mm512_slli_epi32(x2, 8), _mm512_srli_epi32(x1, 16), 0xEA);
                u[3 * g + 2] = _mm512_ternarylogic_epi32(m8, x3, _mm512_srli_epi32(x2, 24), 0xEA);
            }
            const float dxs = dxt[4 * i + ((qr[i >> 2] >> (2 * (i & 3))) & 3)];
            const __m512i z = _mm512_setzero_si512();
            auto sums = [&](int t) {
                const int8_t* a8 = q8[t] + 128 * i;
                const __m512i qa = _mm512_load_si512(a8), qb = _mm512_load_si512(a8 + 64);
                __m512i A = _mm512_dpbusd_epi32(z, u[0], qa);
                A = _mm512_dpbusd_epi32(A, u[1], qa);
                A = _mm512_dpbusd_epi32(A, u[2], qa);
                __m512i B = _mm512_dpbusd_epi32(z, u[3], qb);
                B = _mm512_dpbusd_epi32(B, u[4], qb);
                B = _mm512_dpbusd_epi32(B, u[5], qb);
                return halves8(A, B);
            };
            for (int p = 0; p < np; ++p) {
                const int t = 2 * p;
                const __m256i z0 = sums(t), z1 = sums(t + 1);
                const __m256i s2 = _mm256_add_epi32(_mm256_permute2x128_si256(z0, z1, 0x20), _mm256_permute2x128_si256(z0, z1, 0x31));
                const __m256i b2 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_load_si128((const __m128i*) (bs[t] + 4 * i))),
                                                           _mm_load_si128((const __m128i*) (bs[t + 1] + 4 * i)), 1);
                const __m256i s = _mm256_slli_epi32(_mm256_sub_epi32(s2, b2), 6);
                const __m256 d2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(dd[t] + 4 * i)), _mm_loadu_ps(dd[t + 1] + 4 * i), 1);
                const __m256 scl = _mm256_mul_ps(_mm256_set1_ps(dxs), d2);
                acc2[p] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), scl, acc2[p]);
            }
            if (nt & 1) {
                const int t = nt - 1;
                const __m256i zz = sums(t);
                const __m128i s1 = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(zz), _mm256_extracti128_si256(zz, 1)),
                                                 _mm_load_si128((const __m128i*) (bs[t] + 4 * i)));
                const __m128 scl = _mm_mul_ps(_mm_set1_ps(dxs), _mm_loadu_ps(dd[t] + 4 * i));
                acc1 = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_slli_epi32(s1, 6)), scl, acc1);
            }
        }
        const float rb = tq_half_to_float(sc.rowbase[r]);
        for (int p = 0; p < np; ++p) {
            alignas(32) float v[8];
            _mm256_store_ps(v, acc2[p]);
            out[2 * p][r] = (((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f)) * rb;
            out[2 * p + 1][r] = (((v[4] + v[5]) + (v[6] + v[7])) * (1.0f / 8192.0f)) * rb;
        }
        if (nt & 1) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc1);
            out[nt - 1][r] = (((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f)) * rb;
        }
    }
}

// QuantLab (11/12, strata-desktop's lane-wise form, bench_decode's SM 7; STRATA_TQ_BIH3_LANES=1): the same points and
// VPDPBUSD, but the 16 int32 lanes (a state's 4 weights each) are not folded per block: each lane starts at -384 * its 4
// activations' sum (TqAct::c4), goes to float times its sub-block's activation scale (TqAct::d4), and the block's
// d / 128 accumulates into a ZMM of floats folded once per row.  Another float summation order: not bitwise the
// scalar kernel's (the float rounding only).
template <int KB>
void rows_r1_bih3_lanes_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out,
                          int r0, int r1, const TqScales& sc) {
    const int nb = n / 128;
    constexpr size_t bsz = 4 * KB;
    static const Bih3States<KB> S;
    const __m512i ix0 = _mm512_load_si512(S.idx[0]), ix1 = _mm512_load_si512(S.idx[1]);
    const __m512i sh0 = _mm512_load_si512(S.sh[0]), sh1 = _mm512_load_si512(S.sh[1]);
    const __m512i m15 = _mm512_set1_epi32(0x7FFF);
    const __mmask64 lm = (__mmask64) (bsz >= 64 ? ~0ull : (1ull << bsz) - 1);
    const __m512i c0 = _mm512_set1_epi32((int) 0x9E3779B1u), c1 = _mm512_set1_epi32((int) 0x85EBCA6Bu);
    const __m512i c2 = _mm512_set1_epi32((int) 0xC2B2AE35u), c3 = _mm512_set1_epi32((int) 0x27D4EB2Fu);
    const __m512i m24 = _mm512_set1_epi32((int) 0xFF000000u), m16 = _mm512_set1_epi32((int) 0xFFFF0000u);
    const __m512i m8 = _mm512_set1_epi32((int) 0xFFFFFF00u);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        __m512 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm512_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const __m512i blk = _mm512_maskz_loadu_epi8(lm, x + (size_t) i * bsz);
            __m512i u[6];
            for (int g = 0; g < 2; ++g) {
                const __m512i s = _mm512_and_si512(
                    _mm512_srlv_epi32(_mm512_permutexvar_epi8(g ? ix1 : ix0, blk), g ? sh1 : sh0), m15);
                const __m512i x0 = _mm512_mullo_epi32(s, c0), x1 = _mm512_mullo_epi32(s, c1);
                const __m512i x2 = _mm512_mullo_epi32(s, c2), x3 = _mm512_mullo_epi32(s, c3);
                u[3 * g] = _mm512_ternarylogic_epi32(m24, _mm512_slli_epi32(x1, 16), _mm512_srli_epi32(x0, 8), 0xEA);
                u[3 * g + 1] = _mm512_ternarylogic_epi32(m16, _mm512_slli_epi32(x2, 8), _mm512_srli_epi32(x1, 16), 0xEA);
                u[3 * g + 2] = _mm512_ternarylogic_epi32(m8, x3, _mm512_srli_epi32(x2, 24), 0xEA);
            }
            const __m512 dw = _mm512_set1_ps(tq_half_to_float(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3] *
                                             (1.0f / 128.0f));
            for (int t = 0; t < nt; ++t) {
                const int8_t* a8 = a[t]->q8 + 128 * i;
                const int32_t* cz = a[t]->c4 + 32 * i;
                const float* dz = a[t]->d4 + 32 * i;
                const __m512i qa = _mm512_load_si512(a8), qb = _mm512_load_si512(a8 + 64);
                __m512i A = _mm512_dpbusd_epi32(_mm512_load_si512(cz), u[0], qa);
                A = _mm512_dpbusd_epi32(A, u[1], qa);
                A = _mm512_dpbusd_epi32(A, u[2], qa);
                __m512i B = _mm512_dpbusd_epi32(_mm512_load_si512(cz + 16), u[3], qb);
                B = _mm512_dpbusd_epi32(B, u[4], qb);
                B = _mm512_dpbusd_epi32(B, u[5], qb);
                __m512 f = _mm512_mul_ps(_mm512_cvtepi32_ps(A), _mm512_load_ps(dz));
                f = _mm512_fmadd_ps(_mm512_cvtepi32_ps(B), _mm512_load_ps(dz + 16), f);
                acc[t] = _mm512_fmadd_ps(f, dw, acc[t]);
            }
        }
        const float rb = tq_half_to_float(sc.rowbase[r]);
        for (int t = 0; t < nt; ++t) out[t][r] = _mm512_reduce_add_ps(acc[t]) * rb;
    }
}

// QuantLab (11/4): the AVX2 rows' VNNI copy with the EVEX VPDPWSSD ymm (AVX512-VL/VNNI), so this CPU class tests
// the AVX-VNNI code of tq_avx2_rows.inl (STRATA_TQ_ISA=avx2e; never chosen by default)
namespace evex {
#define STRATA_TQ_VNNI 2
#define STRATA_TQ_ROWS_FN
#include "tq_avx2_rows.inl"
}  // namespace evex

}  // namespace

void tq_rows_avx512(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1) {
    STRATA_TQ_DISPATCH_K(kb, rows_t, table, w, row_bytes, n, a, nt, out, r0, r1)
}
void tq_rows_avx512_r1q2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                         int nt, float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, rows_r1_t, table, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_avx512_r1q2_bih3(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                              float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, rows_r1_bih3_t, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_avx512_r1q2_bih3_pre(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                                  float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, rows_r1_bih3_pre_t, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_avx512_r1q2_bih3_lanes(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                                    float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, rows_r1_bih3_lanes_t, w, row_bytes, n, a, nt, out, r0, r1, sc)
}

void tq_rows_avx2e(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                   float* const* out, int r0, int r1) {
    STRATA_TQ_DISPATCH_K(kb, evex::rows_t, table, w, row_bytes, n, a, nt, out, r0, r1)
}
void tq_rows_avx2e_r1q2(int kb, const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                        int nt, float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, evex::rows_r1_t, table, w, row_bytes, n, a, nt, out, r0, r1, sc)
}
void tq_rows_avx2e_r1q2_bih3(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                             float* const* out, int r0, int r1, const TqScales& sc) {
    STRATA_TQ_DISPATCH_K(kb, evex::rows_r1_bih3_t, w, row_bytes, n, a, nt, out, r0, r1, sc)
}

}  // namespace strata::kernels::cpu::tq_detail
