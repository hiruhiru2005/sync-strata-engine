// third_party/ggml/tq_lut_registry.h - QuantLab (11/4): where the device copies of the trellis codebook are.
//
// ggml-common-tq.h (tq_lut_f16, 2048 fp16 pairs) and src/kernels/cuda/tq_lut_i8.cuh (tq_lut_i8, its int8 copy) are
// `static const __device__` tables: every CUDA translation unit that includes them holds its own copy.  A pack may
// carry its own codebook (tq_lut_f16.bin, see strata/kernels/tq_lut.hpp); then each of those copies has to be
// overwritten before the first kernel runs.  Each such translation unit registers one upload here at static
// initialisation (the headers do it, CUDA only); strata::kernels::tq_lut_device_upload calls them all.
// Header-only on purpose: the overlay of the ggml-cuda headers copies it next to the two tables.
#pragma once

#include <cstdint>
#include <vector>

namespace strata_tq_lut {

/// Copies f16 (4096 u16) or i8 (2048 u16) into this translation unit's table; returns a cudaError_t as int.
using Upload = int (*)(const uint16_t* f16, const uint16_t* i8);

inline std::vector<Upload>& uploads() {
    static std::vector<Upload> v;
    return v;
}

struct Register {
    explicit Register(Upload u) { uploads().push_back(u); }
};

/// QuantLab (11/12): the MMQ tile loaders' bih3 switch (third_party/ggml-tq/tq.cuh, tq_mmq_bih3): sets this translation
/// unit's flag; returns a cudaError_t as int.  strata::kernels::tq_mmq_bih3_upload calls them all.
using FlagSet = int (*)(uint32_t on);
inline std::vector<FlagSet>& bih3_sets() {
    static std::vector<FlagSet> v;
    return v;
}
struct RegisterBih3 {
    explicit RegisterBih3(FlagSet f) { bih3_sets().push_back(f); }
};

}  // namespace strata_tq_lut
