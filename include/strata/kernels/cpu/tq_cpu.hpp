// include/strata/kernels/cpu/tq_cpu.hpp - the trellis expert formats (TQ2_T / TQK6 / TQK7, ggml types 144-146 of the
// agention llama.cpp fork) on the CPU: decode-and-dot against Q8_0 activations (ggml's block_q8_0), one or several
// tokens per call, on AVX-512, AVX2 or plain C++.
//
// THE FORMAT (third_party/ggml/ggml-common-tq.h): a 128-weight block is {fp16 d; 32 steps of K bits} (K = 8 / 6 / 7).
// Step t has a 15-bit state; the state's hash picks two points (pairs) of a 2048-point fp16 codebook, which are
// weights 4t..4t+3, times d.
//
// THE DECODE: a 32768 x 4 int16 table maps a state straight to its four weights at scale 2^13 (256 KiB, the
// approach of the fork's ggml-cpu vec_dot): one 8-byte lookup per four weights, no hash.  The int16 rounding is
// exact for codebook values >= 2^-3 in magnitude and within 2^-14 below that; it is the same table, rounded the
// same way, as the fork's ggml-cpu kernels.
//
// THE ARITHMETIC, identical on every ISA so every path gives the same bits (and so does any token-group size):
//   S[i][k] = sum_j w16[i][32k + j] * q8[i][k][j]        exact in int32 (|w16| < 2^15, |q8| <= 127)
//   acc[k]  = fma((float) S[i][k], d_w[i] * d_a[i][k], acc[k])      for blocks i in order, k = 0..3
//   dot     = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * 2^-13
// i.e. four float accumulators, one per Q8_0 sub-block position, each a fixed sequential chain.  It differs from
// ggml-cpu's vec_dot (same integers, other float summation order) by float rounding only.
//
// THE KERNELS: `tq_isa()` picks AVX-512 (F/BW/VL/VBMI/VNNI, cpu_avx512_ok) or AVX2 (cpu_avx2_ok) at run time; the
// scalar kernel is the reference (on a CPU without AVX2 the engine keeps ggml-cpu's vec_dot, see tq_kernels_enabled).
// STRATA_TQ_ISA=scalar|avx2|avx512 caps the choice (tests, A/B); STRATA_NO_TQ_KERNELS=1 makes native_expert.cpp use
// ggml-cpu's vec_dot instead (the previous path).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// QuantLab (11/4): avx2vnni - the AVX2 rows with AVX-VNNI's VEX VPDPWSSD (cpu_avxvnni_ok: Alder Lake / Raptor Lake and
/// later, Zen 5), chosen by default on a CPU with AVX-VNNI and without AVX-512; avx2e - the same rows with the EVEX
/// VPDPWSSD ymm of AVX512-VL/VNNI, only when STRATA_TQ_ISA=avx2e asks (to test the VNNI copy on an AVX-512 CPU).
/// STRATA_TQ_ISA=scalar|avx2|avxvnni|avx2e|avx512: avx2 is the plain AVX2 rows (no VNNI) even where AVX-VNNI exists.
enum class TqIsa : int { scalar = 0, avx2 = 1, avx512 = 2, avx2vnni = 3, avx2e = 4 };

/// EXPERIMENTAL lower-rate versions of the TQK rule (tools/trellis/tqenc.hpp, version 1; Strata-internal ids, not ggml
/// types, not in any GGUF): TQK5X = 22 bytes / 128 weights (1.375 bpw), TQK4X = 18 bytes (1.125 bpw).  Same hash,
/// codebook, 15-bit mask, LSB-first circular stream and fp16 scale as TQK6/7 with K = 5 / 4.
inline constexpr int kTqTypeK5X = 224;   ///< version 2: K = 5 with the computed codebook (tq_computed.hpp)
inline constexpr int kTqTypeK4X = 225;   ///< version 2: K = 4, computed codebook
inline constexpr int kTqTypeK5T = 226;   ///< version 1: K = 5 with the fork's point table (the first pilots)
inline constexpr int kTqTypeK4T = 227;   ///< version 1: K = 4, point table
/// QuantLab: local-only trellis types kTqPseudoBase + K (K = 2, 3, 9..14) for the rates the Gyro SPEC has no id for.
inline constexpr int kTqPseudoBase = 160;
/// QuantLab (11/11, 11/12): the bih3 types kTqBih3Base + K (K = 2..14): the table types' blocks and state windows, but a
/// step's 4 weights computed from its 15-bit state (no codebook): x_i = s * C_i, weight j = (byte j of T0 + T1 + T2 -
/// 384) / 128 (iq_kernels.cu's tq_bih3p).  Here they run on the same kernels with their own state table (exact: x 64).
inline constexpr int kTqBih3Base = 180;
inline constexpr bool tq_bih3_type(int t) noexcept { return t >= kTqBih3Base + 2 && t <= kTqBih3Base + 14; }
/// The state table of the bih3 types (32768 x 4 int16, the weights x 2^13 like the others).
const int16_t* tq_state_table_bih3();
/// TQ2_T (144), TQK6 (145), TQK7 (146), TQP5/6/7 (147..149), the experimental TQK5X / TQK4X / TQK5T / TQK4T, and QuantLab's
/// TQK5 (150), TQK4 (151) and pseudo types kTqPseudoBase + K.
bool tq_supported(int ggml_type) noexcept;
/// Bits per step: 8, 6, 7, 5, 4, or K of a QuantLab pseudo type (0 for any other type).
int tq_kbits(int ggml_type) noexcept;
/// Values per block of the trellis types, and bytes per block (2 + 4 * bits: 34, 26, 30, 22, 18).
constexpr int kTqBlock = 128;
// TQ2_T / TQK6 / TQK7: 4 weights per step (2 + 4K bytes per 128-value block).  TQP5/6/7 (147-149, Gyro-L) are the
// pair layout: 2 weights per step, 8K bytes per block (2 + 8K).
inline bool tq_pair_layout(int ggml_type) noexcept { return ggml_type >= 147 && ggml_type <= 149; }
inline size_t tq_block_bytes(int ggml_type) noexcept {
    return (size_t) (2 + (tq_pair_layout(ggml_type) ? 8 : 4) * tq_kbits(ggml_type));
}
/// The best ISA this CPU runs (capped by STRATA_TQ_ISA).
TqIsa tq_isa();
/// Whether this CPU can run `isa` at all (scalar: always).
bool tq_isa_ok(TqIsa isa);
/// Whether STRATA_NO_TQ_KERNELS=1 is absent.  The engine's native expert rows take these kernels for the trellis types
/// on every CPU (AVX-512 or AVX2 where the CPU has them, else the scalar kernel: a ggml from upstream has no trellis
/// type to fall back to, and the AVX code is only reached behind tq_isa_ok).  STRATA_NO_TQ_KERNELS=1 asks for
/// ggml-cpu's vec_dot instead, honoured only by a ggml that has the types (the agentionai fork; tests, A/B).
bool tq_kernels_enabled();
const char* tq_isa_name(TqIsa isa);

/// The 32768 x 4 int16 state table of the fork's types (filled on first use; thread-safe).
const int16_t* tq_state_table();
/// The same for the computed codebook of TQK5X / TQK4X (tq_computed.hpp).
const int16_t* tq_state_table_computed();
/// The table a type decodes with.
const int16_t* tq_state_table_for(int ggml_type);

/// Rows [r0, r1) of a trellis matrix (`row_bytes` per row, `n` weights per row) against `nt` (<= 8) Q8_0
/// activations of n values: out[t][r] = w_r . a[t].
void tq_rows(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
             float* const* out, int r0, int r1);
/// Gate and up rows [r0, r1) (gate at blob, up at blob + up_off): ff[t][r] = silu(g) * u, the SwiGLU expression
/// of native_gu_rows.
void tq_gu_rows(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                const void* const* act, int nt, float* const* ff, int r0, int r1);
namespace tq_detail { struct TqScales; }   // QuantLab (10/31): defined below
// QuantLab (10/31): the same rows for R1Q2 blocks (4K bytes each, no fp16 d; tq_detail::TqScales per matrix).  The pair
// layout (147-149) has no R1Q2 form.  Every ISA bitwise equal to the scalar R1Q2 kernel (tqp_cpu_test).
void tq_rows_r1q2(TqIsa isa, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                  float* const* out, int r0, int r1, const tq_detail::TqScales& sc);
void tq_gu_rows_r1q2(TqIsa isa, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n,
                     const void* const* act, int nt, float* const* ff, int r0, int r1, const tq_detail::TqScales& g,
                     const tq_detail::TqScales& u);

/// Largest token group per call and widest row the kernels take.
inline constexpr int kTqMaxTokens = 8;
inline constexpr int kTqMaxN = 5120;   // (DeepSeek-V4.1 hidden width; was 4096)

namespace tq_detail {
/// One token's Q8_0 activation unpacked for the kernels: the int8 values widened to int16 and the scales as floats.
struct TqAct {
    alignas(64) int16_t q[kTqMaxN];
    alignas(64) float d[kTqMaxN / 32];
    // QuantLab (11/12): the bih3 rows' side (tq_rows_avx512_r1q2_bih3): the int8 values as they are, and per Q8_0
    // sub-block 384 * its sum (the planes' -128 three times)
    alignas(64) int8_t q8[kTqMaxN];
    alignas(64) int32_t bsum[kTqMaxN / 32];
    // STRATA_TQ_BIH3_LANES=1 (the lane-wise rows): per 4 values -384 * their sum, and their sub-block's scale
    alignas(64) int32_t c4[kTqMaxN / 4];
    alignas(64) float d4[kTqMaxN / 4];
};
void unpack_act(const void* q8_0, int n, TqAct& a);
/// QuantLab (10/31): one matrix's R1Q2 scales.  With them a block is its 4K code bytes only (no fp16 d): the block's
/// scale is colscale[b] * p2[q[r][b]] and each row's dot is multiplied by rowbase[r] once at the end.
struct TqScales {
    const uint16_t* rowbase = nullptr;    // fp16 [rows]
    const uint16_t* colscale = nullptr;   // fp16 [n / 128]
    const uint8_t* q = nullptr;           // [rows][qbytes]: block b's 2-bit code at bits 2 * (b % 4) of byte b / 4
    int qbytes = 0;
    float p2[4] = {0.f, 0.f, 0.f, 0.f};   // 2^(lo + q * step), q = 0..3
};
void tq_rows_scalar_r1q2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                         int nt, float* const* out, int r0, int r1, const TqScales& sc);
void tq_rows_avx2_r1q2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                       int nt, float* const* out, int r0, int r1, const TqScales& sc);
void tq_rows_avx512_r1q2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                         int nt, float* const* out, int r0, int r1, const TqScales& sc);
/// QuantLab (11/12): the R1Q2 rows of the bih3 types computed (no table): 16 states a ZMM, x_i by VPMULLD, the three
/// byte planes by VPTERNLOG, VPDPBUSD against the int8 activations, minus 384 * the sub-block sums: the scalar kernel's
/// exact S[k] (/ 64), so bitwise equal to it.  STRATA_TQ_BIH3_TABLE=1 keeps the table rows instead (A/B).
void tq_rows_avx512_r1q2_bih3(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                              float* const* out, int r0, int r1, const TqScales& sc);
/// QuantLab (11/12): STRATA_TQ_BIH3_LANES=1 - the same, the 16 lanes kept apart in floats until the row's end (another
/// summation order: equal to the scalar kernel to the float rounding, not bitwise).
void tq_rows_avx512_r1q2_bih3_lanes(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                                    float* const* out, int r0, int r1, const TqScales& sc);
// QuantLab (11/4): the AVX-VNNI copies of the AVX2 rows (tq_avx2.cpp) and their EVEX test copies (tq_avx512.cpp)
bool tq_avxvnni_built();
void tq_rows_avxvnni(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                     float* const* out, int r0, int r1);
void tq_rows_avxvnni_r1q2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                          int nt, float* const* out, int r0, int r1, const TqScales& sc);
void tq_rows_avx2e(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                   float* const* out, int r0, int r1);
void tq_rows_avx2e_r1q2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a,
                        int nt, float* const* out, int r0, int r1, const TqScales& sc);
// the per-ISA row kernels (dot products only; `out[t][r]` for rows [r0, r1))
void tq_rows_scalar(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
void tq_rows_avx2(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                  float* const* out, int r0, int r1);
/// The TQP pair layout (types 147-149) on AVX2.  AVX-512 also takes this kernel: the pair layout has no
/// AVX-512 form yet (no AVX-512 machine has verified one), and the AVX2 path is exact.
void tq_rows_avx2_tqp(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                      float* const* out, int r0, int r1);
/// The scalar reference of the pair layout (any CPU).
void tq_rows_scalar_tqp(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                        float* const* out, int r0, int r1);
void tq_rows_avx512(int kb, const int16_t* tab, const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt,
                    float* const* out, int r0, int r1);
}  // namespace tq_detail

}  // namespace strata::kernels::cpu
