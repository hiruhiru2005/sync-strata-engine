// include/strata/kernels/tq_lut.hpp - QuantLab (11/4): the trellis codebook the engine decodes with.
//
// The trellis (APR) expert formats map each 15-bit state to two of 2048 codebook points (third_party/ggml/
// ggml-common-tq.h, tq_lut_f16: 2048 pairs of fp16).  The engine holds that table in three forms: the CPU kernels'
// int16 state table (cpu::tq_state_table, round(v * 2^13)), the GPU's fp16 copies and the GPU's int8 copy
// (src/kernels/cuda/tq_lut_i8.cuh, one global scale TQ_LUT_I8_SCALE) that decode and the prompt path's MMQ read.
//
// A pack may carry its own codebook, quantized together with its experts: <pack>/tq_lut_f16.bin, the 4096 fp16 bit
// patterns raw (8192 bytes, little-endian, point i = [2i], [2i+1]).  tq_lut_load replaces the host table (the CPU
// table follows it); the caller then uploads it with tq_lut_device_upload, before the first kernel runs.  Without
// the file nothing changes: the built-in table, byte-identical to before.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {

inline constexpr int kTqLutPoints = 2048;
inline constexpr int kTqLutValues = 2 * kTqLutPoints;          // 4096 fp16
inline constexpr std::size_t kTqLutBytes = kTqLutValues * 2;   // 8192
inline constexpr const char* kTqLutFile = "tq_lut_f16.bin";

/// The current codebook (4096 fp16 bit patterns): built-in until tq_lut_load replaced it.
const uint16_t* tq_lut_f16_host();
/// Whether the current codebook is the built-in one (byte for byte).
bool tq_lut_builtin();
/// Bumped by every tq_lut_load that succeeds (the CPU state table refills when it changes).
uint64_t tq_lut_generation();
/// SHA-256 (lowercase hex) of the current codebook's 8192 bytes, and of the built-in one.
std::string tq_lut_sha256();
std::string tq_lut_builtin_sha256();
/// SHA-256 of a codebook file (which must be 8192 bytes).
bool tq_lut_file_sha256(const std::string& path, std::string& hex, std::string& err);
/// Reads a codebook file and makes it the current one: exactly 8192 bytes, every value finite and within the int8
/// copy's range (|v| <= 127 * TQ_LUT_I8_SCALE = 3.984375, which also keeps the CPU's round(v * 2^13) in int16).
/// Also checks that the int8 generator reproduces the built-in int8 table from the built-in codebook.  Call it
/// before any kernel runs; nothing else is changed when it fails.
bool tq_lut_load(const std::string& path, std::string& err);
/// The int8 copy of a codebook (2048 u16: point i = lo byte, hi byte), as tq_lut_i8.cuh holds the built-in one.
void tq_lut_i8_from_f16(const uint16_t* f16, uint16_t* i8);
/// The largest |value| of a codebook.
float tq_lut_max_abs(const uint16_t* f16);

/// Overwrites every registered device copy (tq_lut_registry.h) on the CURRENT device with f16 / its int8 copy and
/// synchronizes.  copies: how many tables were written.  CUDA builds only (HIP: false, with an error).
bool tq_lut_device_upload(const uint16_t* f16, const uint16_t* i8, int& copies, std::string& err);

}  // namespace strata::kernels
