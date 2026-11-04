// src/kernels/cpu/tq_avx2_rows.inl - the AVX2 trellis rows (TQ2_T / TQK, and QuantLab's R1Q2 blocks) of tq_avx2.cpp,
// included more than once, each time inside a namespace of its own:
//   tq_avx2.cpp   plain     STRATA_TQ_VNNI 0: VPMADDWD, VPMADDWD, VPADDD per pair of weight registers
//   tq_avx2.cpp   vnni      STRATA_TQ_VNNI 1: VPMADDWD then the VEX VPDPWSSD of AVX-VNNI (Alder Lake / Raptor Lake and
//                           later, P- and E-cores; Zen 5) - taken where cpu_avxvnni_ok()
//   tq_avx512.cpp evex      STRATA_TQ_VNNI 2: the same with the EVEX VPDPWSSD ymm of AVX512-VL/VNNI, so an AVX-512 CPU
//                           without AVX-VNNI (Zen 4) runs and tests the VNNI code (STRATA_TQ_ISA=avx2e)
// The int32 sums are exact either way (|w16| < 2^15, |q8| <= 127), so every copy gives the scalar kernel's bits.
// Before including: STRATA_TQ_VNNI and STRATA_TQ_ROWS_FN (nothing, or GCC/Clang's target("avxvnni") for copy 1).
//
// QuantLab (11/4): the 32 states of a block are built with compile-time indices (tq_state_c, rows4c).  The loop over
// them carried `#pragma GCC unroll 8`, which MSVC ignores: on the Windows build it stayed a loop, every state's
// offset, branch and shift ran at run time, and the AVX2 rows took ~40 ns per block against ~22 ns for the AVX-512
// rows (whose states were constants already) on the same CPU.

#if defined(_MSC_VER)
#define STRATA_TQ_INLINE __forceinline
#else
#define STRATA_TQ_INLINE inline __attribute__((always_inline))
#endif

// four state-table rows (8 bytes each) into one YMM: a load and three broadcast loads blended in.  Broadcast loads
// and VPBLENDD avoid the shuffle port that VPINSRQ / VINSERTI128 need (the two measure the same on Zen 5).
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i rows4(const long long* tab, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    __m256i v = _mm256_castsi128_si256(_mm_loadl_epi64((const __m128i*) (tab + s0)));
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s1]), 0x0C);
    v = _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s2]), 0x30);
    return _mm256_blend_epi32(v, _mm256_set1_epi64x(tab[s3]), 0xC0);
}

// weights 16M .. 16M+15 of a block (states 4M .. 4M+3), the states compile-time
template <int KB, int M>
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i rows4c(const long long* tab, const uint8_t* qs, uint64_t wr) {
    return rows4(tab, tq_state_c<KB, 4 * M>(qs, wr), tq_state_c<KB, 4 * M + 1>(qs, wr), tq_state_c<KB, 4 * M + 2>(qs, wr),
                 tq_state_c<KB, 4 * M + 3>(qs, wr));
}

// (wa . qa) + (wb . qb) in int32 pairs: the partial sums of one Q8_0 sub-block (32 weights)
STRATA_TQ_ROWS_FN STRATA_TQ_INLINE __m256i dot2(__m256i wa, __m256i qa, __m256i wb, __m256i qb) {
#if STRATA_TQ_VNNI == 1
    return _mm256_dpwssd_avx_epi32(_mm256_madd_epi16(wa, qa), wb, qb);
#elif STRATA_TQ_VNNI == 2
    return _mm256_dpwssd_epi32(_mm256_madd_epi16(wa, qa), wb, qb);
#else
    return _mm256_add_epi32(_mm256_madd_epi16(wa, qa), _mm256_madd_epi16(wb, qb));
#endif
}

// S_ = one block's four exact sub-block sums S[0..3] against one token's unpacked activations (128 int16 at QP_).
// Statements, not a lambda: a lambda would not carry GCC's target("avxvnni") of the VNNI copy.
#define STRATA_TQ_SUMS4(S_, QP_)                                                                                       \
    const __m256i* S_##q = (const __m256i*) (QP_);                                                                    \
    const __m256i S_##p0 = dot2(w0, _mm256_load_si256(S_##q + 0), w1, _mm256_load_si256(S_##q + 1));                \
    const __m256i S_##p1 = dot2(w2, _mm256_load_si256(S_##q + 2), w3, _mm256_load_si256(S_##q + 3));                \
    const __m256i S_##p2 = dot2(w4, _mm256_load_si256(S_##q + 4), w5, _mm256_load_si256(S_##q + 5));                \
    const __m256i S_##p3 = dot2(w6, _mm256_load_si256(S_##q + 6), w7, _mm256_load_si256(S_##q + 7));                \
    /* per 128-bit lane [p0, p1, p2, p3] partial sums, then the two lanes added: S[0..3], exact */                  \
    const __m256i S_##h = _mm256_hadd_epi32(_mm256_hadd_epi32(S_##p0, S_##p1), _mm256_hadd_epi32(S_##p2, S_##p3));  \
    const __m128i S_ = _mm_add_epi32(_mm256_castsi256_si128(S_##h), _mm256_extracti128_si256(S_##h, 1))

#define STRATA_TQ_WEIGHTS8(tab_, qs_, wr_)                                                                                 \
    const __m256i w0 = rows4c<KB, 0>(tab_, qs_, wr_), w1 = rows4c<KB, 1>(tab_, qs_, wr_);                                 \
    const __m256i w2 = rows4c<KB, 2>(tab_, qs_, wr_), w3 = rows4c<KB, 3>(tab_, qs_, wr_);                                 \
    const __m256i w4 = rows4c<KB, 4>(tab_, qs_, wr_), w5 = rows4c<KB, 5>(tab_, qs_, wr_);                                 \
    const __m256i w6 = rows4c<KB, 6>(tab_, qs_, wr_), w7 = rows4c<KB, 7>(tab_, qs_, wr_)

template <int KB>
STRATA_TQ_ROWS_FN void rows_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                              float* const* out, int r0, int r1) {
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
            STRATA_TQ_WEIGHTS8(tab, qs, wr);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                STRATA_TQ_SUMS4(s, a[t]->q + 128 * i);
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
STRATA_TQ_ROWS_FN void rows_r1_t(const int16_t* table, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                                 int nt, float* const* out, int r0, int r1, const TqScales& sc) {
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
            STRATA_TQ_WEIGHTS8(tab, qs, wr);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3]);
            for (int t = 0; t < nt; ++t) {
                STRATA_TQ_SUMS4(s, a[t]->q + 128 * i);
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

#undef STRATA_TQ_WEIGHTS8
#undef STRATA_TQ_SUMS4
#undef STRATA_TQ_INLINE
#undef STRATA_TQ_ROWS_FN
#undef STRATA_TQ_VNNI
