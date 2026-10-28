// include/strata/kernels/tq_mix.hpp - QuantLab (local builds only): per-expert trellis K (native_experts.txt v6,
// experts_k.txt).  A MIXED layer's experts each have their own trellis rate: K gate/up and K down, each 2..12.
//
// The GPU's grouped launch of such a layer takes NativeExpertLayout::gu_type = d_type = kTqMixedType, and each group's
// blob pointer carries its expert's (K gate/up, K down) in the pointer's top byte (tq_mix_tag): the kernels switch per
// group to that K's rows.  Device addresses and pinned host aliases are below 2^48, so the byte is free; the host plan
// checks it before tagging.  No dependency: included by the CUDA kernels and by the CPU-side planner alike.
#pragma once

#include <cstdint>

namespace strata::kernels {

inline constexpr int kTqMixedType = 159;   ///< NativeExpertLayout gu_type / d_type of a mixed layer's launch
inline constexpr int kTqTagShift = 56;
inline constexpr unsigned long long kTqAddrMask = (1ull << kTqTagShift) - 1;

/// The tag of an expert with gate/up rate `k_gu` and down rate `k_d`: K gate/up in bits 56..59, K down in 60..63.
inline constexpr unsigned long long tq_mix_tag(int k_gu, int k_d) {
    return ((unsigned long long) ((k_gu & 15) | ((k_d & 15) << 4))) << kTqTagShift;
}
/// Whether the mixed-layer kernels take this pair (every K 2..12 has a row kernel).
inline constexpr bool tq_mix_supported(int k_gu, int k_d) { return k_gu >= 2 && k_gu <= 14 && k_d >= 2 && k_d <= 14; }   // 10/26: K 13, 14 added

}  // namespace strata::kernels
