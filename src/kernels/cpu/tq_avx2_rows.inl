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

// ---- QuantLab (11/12): the bih3 R1Q2 rows (types 180 + K), the points computed (no table), bitwise the scalar kernel's.
// States: per group of 4 (t = 4m .. 4m+3) a 16-byte window of the block (a direct load, or for the windows that wrap
// around the circular stream, bytes from the block's first and last 16), VPSHUFB to the 4 bytes of each state's dword,
// VPSRLVD, the 15-bit mask; two groups make a YMM of 8 states = 32 weights = one Q8_0 sub-block.  Points: x_i = s * C_i
// (VPMULLD), then per state the two int16 words U1 = [v0, v1], U2 = [v2, v3] (v_j = T0.bj + T1.bj + T2.bj, 0..765) by
// VPSHUFB of the x_i's bytes and adds; VPMADDWD against the activations split in pairs (TqAct::qp0 = [a0, a1],
// qp1 = [a2, a3] per state; VPDPWSSD on the VNNI copies), the 8 lanes summed per sub-block: sum_j v_j * a_j exactly;
// minus 384 * the sub-block's sum (TqAct::bsum), x 64 = the table rows' S[k], and the scalar kernel's float steps.
template <int KB>
struct Bih3Win {
    int dir[8];                      // group m's direct load byte, -1: from F (the first 16 bytes) and E (the last 16)
    alignas(16) uint8_t idx[8][16];  // the shuffle of the direct window, or of E
    alignas(16) uint8_t idxf[8][16]; // the shuffle of F (for the groups that are not direct)
    alignas(32) uint32_t sh[4][8];   // per YMM g (groups 2g, 2g+1): the right shifts
    Bih3Win() {
        constexpr int NB = 4 * KB;
        for (int m = 0; m < 8; ++m) {
            int need[4][4];   // state i's 4 bytes (unwrapped stream indices)
            for (int i = 0; i < 4; ++i) {
                const int t = 4 * m + i;
                if (KB == 8) {
                    need[i][0] = t; need[i][1] = (t + 31) & 31; need[i][2] = need[i][3] = -1;
                    sh[m >> 1][4 * (m & 1) + i] = 0;
                } else {
                    const int off = (31 - t) * KB, b = off >> 3;
                    for (int j = 0; j < 4; ++j) need[i][j] = b + j;
                    sh[m >> 1][4 * (m & 1) + i] = (uint32_t) (off & 7);
                }
            }
            int lo = 1 << 30, hi = -1;
            for (auto& r : need)
                for (int v : r)
                    if (v >= 0) { lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
            dir[m] = (NB >= 16 && hi < NB && hi - lo < 16 && lo + 16 <= NB) ? lo : -1;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    const int v = need[i][j];
                    uint8_t& e = idx[m][4 * i + j];
                    uint8_t& f = idxf[m][4 * i + j];
                    e = f = 0x80;
                    if (v < 0) continue;
                    if (dir[m] >= 0) { e = (uint8_t) (v - dir[m]); continue; }
                    const int w = v % NB;
                    if (NB < 16 || w < 16) f = (uint8_t) w;          // from F (the whole block when it is short)
                    else if (w >= NB - 16) e = (uint8_t) (w - (NB - 16));
                    else std::abort();                                 // a wrapped window with a middle byte: never
                }
        }
    }
};

template <int KB>
STRATA_TQ_ROWS_FN void rows_r1_bih3_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                                      float* const* out, int r0, int r1, const TqScales& sc) {
    const int nb = n / 128;
    constexpr int NB = 4 * KB;
    static const Bih3Win<KB> W;
    const __m256i m15 = _mm256_set1_epi32(0x7FFF);
    const __m256i c0 = _mm256_set1_epi32((int) 0x9E3779B1u), c1 = _mm256_set1_epi32((int) 0x85EBCA6Bu);
    const __m256i c2 = _mm256_set1_epi32((int) 0xC2B2AE35u), c3 = _mm256_set1_epi32((int) 0x27D4EB2Fu);
    // per dword lane d: U1 = [x0.b1 + x1.b2 + x2.b3 | x0.b2 + x1.b3 + x3.b1], U2 = [x0.b3 + x2.b1 + x3.b2 | x1.b1 + x2.b2 + x3.b3]
    // (each term one VPSHUFB of one x_i: bytes to positions 0 / 2 of the dword, zeros elsewhere)
    const __m256i u1a = _mm256_set1_epi32((int) 0x80028001u), u1b = _mm256_set1_epi32((int) 0x80038002u);
    const __m256i u1c = _mm256_set1_epi32((int) 0x80808003u), u1d = _mm256_set1_epi32((int) 0x80018080u);
    const __m256i u2a = _mm256_set1_epi32((int) 0x80808003u), u2b = _mm256_set1_epi32((int) 0x80018080u);
    const __m256i u2c = _mm256_set1_epi32((int) 0x80028001u), u2d = _mm256_set1_epi32((int) 0x80038002u);
    const __m256i lane4 = _mm256_setr_epi8(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8,
                                           12, 12, 12, 12);
    // the dword-relative masks above get + 4 d per lane (0x80 stays >= 0x80: still a zero byte)
    const __m256i U1A = _mm256_add_epi8(u1a, lane4), U1B = _mm256_add_epi8(u1b, lane4), U1C = _mm256_add_epi8(u1c, lane4),
                  U1D = _mm256_add_epi8(u1d, lane4), U2A = _mm256_add_epi8(u2a, lane4), U2B = _mm256_add_epi8(u2b, lane4),
                  U2C = _mm256_add_epi8(u2c, lane4), U2D = _mm256_add_epi8(u2d, lane4);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* qs = x + (size_t) i * NB;
            __m128i F, E;
            if constexpr (NB >= 16) {
                F = _mm_loadu_si128((const __m128i*) qs);
                E = _mm_loadu_si128((const __m128i*) (qs + NB - 16));
            } else {   // K 2 (8 bytes), K 3 (12 bytes): the whole block in F
                F = _mm_loadl_epi64((const __m128i*) qs);
                if constexpr (NB > 8) F = _mm_insert_epi32(F, (int) tq_ld32(qs + 8), 2);
                E = F;
            }
            __m256i U1[4], U2[4];
            for (int g = 0; g < 4; ++g) {
                __m128i s4[2];
                for (int h = 0; h < 2; ++h) {
                    const int m = 2 * g + h;
                    if (W.dir[m] >= 0)
                        s4[h] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*) (qs + W.dir[m])),
                                                 _mm_load_si128((const __m128i*) W.idx[m]));
                    else
                        s4[h] = _mm_or_si128(_mm_shuffle_epi8(E, _mm_load_si128((const __m128i*) W.idx[m])),
                                             _mm_shuffle_epi8(F, _mm_load_si128((const __m128i*) W.idxf[m])));
                }
                const __m256i s = _mm256_and_si256(
                    _mm256_srlv_epi32(_mm256_set_m128i(s4[1], s4[0]), _mm256_load_si256((const __m256i*) W.sh[g])), m15);
                const __m256i x0 = _mm256_mullo_epi32(s, c0), x1 = _mm256_mullo_epi32(s, c1);
                const __m256i x2 = _mm256_mullo_epi32(s, c2), x3 = _mm256_mullo_epi32(s, c3);
                U1[g] = _mm256_add_epi16(_mm256_add_epi16(_mm256_shuffle_epi8(x0, U1A), _mm256_shuffle_epi8(x1, U1B)),
                                         _mm256_add_epi16(_mm256_shuffle_epi8(x2, U1C), _mm256_shuffle_epi8(x3, U1D)));
                U2[g] = _mm256_add_epi16(_mm256_add_epi16(_mm256_shuffle_epi8(x0, U2A), _mm256_shuffle_epi8(x1, U2B)),
                                         _mm256_add_epi16(_mm256_shuffle_epi8(x2, U2C), _mm256_shuffle_epi8(x3, U2D)));
            }
            const __m128 dx = _mm_set1_ps(tq_half_to_float(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3]);
            for (int t = 0; t < nt; ++t) {
                const __m256i* p0 = (const __m256i*) (a[t]->qp0 + 64 * i);
                const __m256i* p1 = (const __m256i*) (a[t]->qp1 + 64 * i);
                const __m256i d0 = dot2(U1[0], _mm256_load_si256(p0 + 0), U2[0], _mm256_load_si256(p1 + 0));
                const __m256i d1 = dot2(U1[1], _mm256_load_si256(p0 + 1), U2[1], _mm256_load_si256(p1 + 1));
                const __m256i d2 = dot2(U1[2], _mm256_load_si256(p0 + 2), U2[2], _mm256_load_si256(p1 + 2));
                const __m256i d3 = dot2(U1[3], _mm256_load_si256(p0 + 3), U2[3], _mm256_load_si256(p1 + 3));
                const __m256i hh = _mm256_hadd_epi32(_mm256_hadd_epi32(d0, d1), _mm256_hadd_epi32(d2, d3));
                const __m128i s1 = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(hh), _mm256_extracti128_si256(hh, 1)),
                                                 _mm_load_si128((const __m128i*) (a[t]->bsum + 4 * i)));
                const __m128 scl = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_slli_epi32(s1, 6)), scl, acc[t]);
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
