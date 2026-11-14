#pragma once
// QuantLab (11/15) STRATA_QL_LAG=1: who waited for whom in a decode window, counted by the GPU itself (no STRATA_VERIFY_PROFILE,
// so the fused paths stay as they are).  Every kernel that spins on a host flag (wait_flag_ge, wait_flag_ge_or,
// wait_flag_ge_stamped, wait_copy_i32_from_mapped, copy_rows_from_mapped_after) adds its spin time by the flag it waited
// on: 0 the host's GPU plan (flag A), 1 the PCIe share (flag B), 2 the CPU experts' rows (the GPU waited for the CPU),
// 3 any other.  The router's publish (doorbell_publish*, doorbell_ring) adds the GPU time since the last CPU-rows wait
// released (the layer's combine + the next layer's dense part up to its router): the host's GPU-reach wait minus this is
// the time the CPU waited for the GPU's experts.  Mapped host memory, read by the host after the window; off = null.
#include <cstdint>
#include <string>

namespace strata::kernels {

struct QlLag {
    unsigned long long ns[4];       // spin ns by kind (0 plan A, 1 PCIe B, 2 CPU rows, 3 other)
    unsigned long long n[4];        // waits by kind
    unsigned long long spun[4];     // waits that spun more than 20 us
    unsigned long long post_ns;     // CPU-rows wait released -> the next router published (GPU clock)
    unsigned long long n_post;
    unsigned long long last_go;     // GPU clock of the last CPU-rows release in this window (0 = none yet)
    unsigned long long fA, fB, fC;  // the device addresses of flag A, flag B and the CPU-rows flag
};

/// The shared counters (host pointer), allocated and handed to the kernels on the first call; null unless STRATA_QL_LAG=1
/// (and on HIP builds).
QlLag* ql_lag_host();
/// The three flags' device addresses (verify's m_flagA_, m_flagB_, m_flag_), so a wait is counted by its kind.
void ql_lag_set_flags(const void* flag_a, const void* flag_b, const void* flag_rows);
/// Before a decode window: forget the last CPU-rows release (the first layer's router follows the draft, not a wait).
void ql_lag_window_begin();

/// QuantLab (11/15) STRATA_QL_PROBE (E-F): the GPU half of the start-up probe - the card, the H2D copies (pinned and
/// pageable) and D2H, the GPU's read of VRAM and of mapped host memory (the PCIe share's path), and the round trip of a
/// host flag to a spinning kernel and back.  JSON members without the braces ("gpu_probe": "skipped" on HIP builds).
std::string ql_probe_gpu();

}  // namespace strata::kernels
