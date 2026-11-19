// strata/core/hip_arena.hpp - QuantLab (11/19) STRATA_HIP_ARENA: how the RAM experts' page-locked copy (the cache
// complement, FileExpertSource) is taken on an AMD card.  HIP builds only (STRATA_USE_HIP / STRATA_HIP_GFX906): the CUDA
// build neither includes this header nor compiles src/core/hip_arena.cu.
//
// The copy is read by the GPU over PCIe every window (--pcie-mode kernel / direct: a kernel reads it through its device
// alias; dma: the copy engine).  It used to be taken one way only, cudaHostAlloc(Mapped | Portable), which HIP makes
// fine-grained (coherent): the GPU does not cache it.  STRATA_HIP_ARENA tries the coarse-grained ways too and keeps the
// fastest one that passes a stale-read test on this card and runtime (see hip_arena.cu for the test and the rule).
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

namespace strata::core::hip_arena {

enum Mode : int {
    kCoherent = 0,     ///< cudaHostAlloc(Mapped | Portable): the allocation as before (fine-grained)
    kNonCoherent = 1,  ///< hipHostMalloc(Portable | NonCoherent): coarse-grained, the device alias of the same block
    kRegister = 2,     ///< page-aligned ordinary memory + hipHostRegister(Mapped | Portable | coarse-grained)
};

/// The way a complement of `bytes` is to be taken.  STRATA_HIP_ARENA=auto (the default) measures a small region taken
/// each way (GPU read, copy engine, CPU read, and a stale-read test) and picks; =coherent keeps the old allocation without
/// measuring; =noncoherent / =register force one.  One stderr line says what was measured and picked.  The first call
/// measures, later calls (a second source) reuse its answer.  Never fails: kCoherent whenever anything is in doubt.
int choose(size_t bytes);

/// In place of cudaHostAlloc(p, bytes, Mapped | Portable): the block taken `mode`'s way with its device alias checked.
/// A refused non-coherent / registered block falls back to the coherent allocation (said on stderr); `taken` (may be
/// null) receives the way that was really used.  The return is the final allocation's status.
hipError_t alloc(void** p, size_t bytes, int mode, int* taken);

/// Frees a block alloc() returned (any cudaHostAlloc'd block too): a registered one is unregistered and released.
void free_arena(void* p);

const char* mode_name(int mode);

}  // namespace strata::core::hip_arena
