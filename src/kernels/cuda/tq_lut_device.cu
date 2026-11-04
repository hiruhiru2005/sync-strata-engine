// src/kernels/cuda/tq_lut_device.cu - QuantLab (11/4): writes a pack's own trellis codebook into every device copy
// (each CUDA translation unit that includes ggml-common-tq.h / tq_lut_i8.cuh registered its tables at static
// initialisation, tq_lut_registry.h).  See strata/kernels/tq_lut.hpp.
#include "strata/kernels/tq_lut.hpp"
#include "tq_lut_registry.h"

#include <cuda_runtime.h>

#include <string>

namespace strata::kernels {

bool tq_lut_device_upload(const uint16_t* f16, const uint16_t* i8, int& copies, std::string& err) {
    copies = 0;
#if defined(STRATA_TQ_MMQ_FORK)
    (void) f16;
    (void) i8;
    err = "this build's prompt-path MMQ comes from the agention llama.cpp fork, which holds its own trellis codebook: "
          "a pack's tq_lut_f16.bin cannot replace it (build with the overlay, STRATA_MMQ_TQ_OVERLAY)";
    return false;
#elif defined(__HIPCC__) || defined(__HIP__)
    (void) f16;
    (void) i8;
    err = "a pack's own trellis codebook (tq_lut_f16.bin) is not supported on HIP yet";
    return false;
#else
    for (::strata_tq_lut::Upload u : ::strata_tq_lut::uploads()) {
        const int rc = u(f16, i8);
        if (rc != 0) {
            err = std::string("trellis codebook upload: ") + cudaGetErrorString((cudaError_t) rc);
            return false;
        }
        ++copies;
    }
    const cudaError_t s = cudaDeviceSynchronize();
    if (s != cudaSuccess) {
        err = std::string("trellis codebook upload: ") + cudaGetErrorString(s);
        return false;
    }
    if (copies == 0) {
        err = "trellis codebook upload: no device copy registered (tq_lut_registry.h)";
        return false;
    }
    return true;
#endif
}

}  // namespace strata::kernels
