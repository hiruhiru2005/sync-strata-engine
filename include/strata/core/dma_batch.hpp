#pragma once
// dma_batch.hpp - independent host-to-device blob copies as one submission: copy_blobs.
//
// The batched path is midhatn's (PR #807, MIT-compatible contribution): cudaMemcpyBatchAsync for independent
// pinned expert uploads on CUDA 13+, with the Windows alias lookup it found necessary.  eddoursul's fork (F12) uses
// the same call for the adaptive tier's and the prefetch's copies: under WDDM a copy submitted alone raises an
// interrupt when it lands, on the processor its GPU's interrupts go to; a batch raises one.
//
// Modes (STRATA_DMA_BATCH, read once; `dma_batch_mode()`):
//   0  one cudaMemcpyAsync per blob: exactly the loop the engine always ran
//   1  cudaMemcpyBatchAsync (CUDA 13+, groups of 2..128), the copies in any order, the batch in the stream's
//   2  as 1, plus CUDA 13.4's cudaMemcpyFlagPreferOverlapWithCompute (a hint)
// QuantLab (11/16): CUDA 12.8-12.9 batches too (12.8's signature) when the driver is 12.8+, and HIP on ROCm 7+
// (hipMemcpyBatchAsync).  Every other build (CUDA before 12.8, older ROCm) and every single copy or group over 128
// takes the loop.  A batch the
// runtime refuses is redone as the loop (with one warning), never dropped: readiness is signalled only for
// uploads that were submitted.
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::core {

/// The mode: STRATA_DMA_BATCH (0, 1, 2); unset = the engine's default (`default_mode`, set at the call site's first
/// use; 0 until a measured win makes it 1).
inline int dma_batch_mode(int default_mode = 0) {
    static const int mode = [default_mode] {
        const char* e = std::getenv("STRATA_DMA_BATCH");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : default_mode;
    }();
    return mode;
}

namespace detail {

// A registered Windows host pointer may differ from its device-visible alias; the batch API needs the alias
// (cudaMemcpyAsync(kind=HostToDevice) resolves it itself).  The experts' blobs are fixed, so each is looked up once.
inline cudaError_t host_alias(const void* p, const void** out) {
    static std::mutex mu;
    static std::unordered_map<const void*, const void*> cache;
    {
        std::lock_guard<std::mutex> lk(mu);
        const auto it = cache.find(p);
        if (it != cache.end()) { *out = it->second; return cudaSuccess; }
    }
    void* alias = nullptr;
    const cudaError_t e = cudaHostGetDevicePointer(&alias, const_cast<void*>(p), 0);
    if (e != cudaSuccess) return e;
    std::lock_guard<std::mutex> lk(mu);
    cache.emplace(p, alias);
    *out = alias;
    return cudaSuccess;
}

}  // namespace detail

/// `n` independent copies, dst[i] <- src[i] (bytes[i] each), on `stream`.  The sources are immutable pinned host
/// blobs and the destinations disjoint; the caller keeps them alive until the stream completes.  Ordering with
/// earlier and later work on the stream is that of the copies one by one.  Returns the first error (the loop
/// continues past none: the first failing copy ends it).
inline cudaError_t copy_blobs(void* const* dst, const void* const* src, const size_t* bytes, size_t n,
                              cudaStream_t stream, int batch_mode) {
    if (n == 0) return cudaSuccess;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    constexpr size_t cap = 128;
    if (batch_mode != 0 && n > 1 && n <= cap) {
        std::array<const void*, cap> srcs{};
        bool aliased = true;
        for (size_t i = 0; i < n; ++i)
            if (detail::host_alias(src[i], &srcs[i]) != cudaSuccess) { aliased = false; break; }
        if (aliased) {
            cudaMemcpyAttributes attr{};
            attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
#if CUDART_VERSION >= 13040
            if (batch_mode == 2) attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
#endif
            size_t first = 0;
            std::array<void*, cap> dsts{};
            std::array<size_t, cap> sizes{};
            for (size_t i = 0; i < n; ++i) { dsts[i] = dst[i]; sizes[i] = bytes[i]; }
            const cudaError_t e = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), n, &attr, &first, 1, stream);
            if (e == cudaSuccess) return cudaSuccess;
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "strata: cudaMemcpyBatchAsync failed (%s); the uploads go one copy at a time\n",
                             cudaGetErrorString(e));
            }
            (void) cudaGetLastError();
        }
    }
#elif !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
    // QuantLab (11/16): CUDA 12.8-12.9 (the GTX 10 build) batches too: the same call with 12.8's signature (non-const
    // sources and the failing index).  The 12.x runtime also runs on drivers older than 12.8 (minor-version
    // compatibility): there the call is not available, so the driver version is read once and a batch is only tried on
    // a 12.8+ driver; a refusal that says the driver or the device cannot do it turns the batch off for the run (one
    // warning), any other refusal is redone as the loop as on CUDA 13.
    constexpr size_t cap = 128;
    static const bool driver_ok = [] {
        int v = 0;
        return cudaDriverGetVersion(&v) == cudaSuccess && v >= 12080;
    }();
    static std::atomic<bool> batch_off{false};
    if (batch_mode != 0 && driver_ok && !batch_off && n > 1 && n <= cap) {
        std::array<void*, cap> srcs{};
        bool aliased = true;
        for (size_t i = 0; i < n; ++i) {
            const void* a = nullptr;
            if (detail::host_alias(src[i], &a) != cudaSuccess) { aliased = false; break; }
            srcs[i] = const_cast<void*>(a);
        }
        if (aliased) {
            cudaMemcpyAttributes attr{};
            attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
            if (batch_mode == 2) attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
            size_t first = 0, fail = 0;
            std::array<void*, cap> dsts{};
            std::array<size_t, cap> sizes{};
            for (size_t i = 0; i < n; ++i) { dsts[i] = dst[i]; sizes[i] = bytes[i]; }
            const cudaError_t e = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), n, &attr, &first, 1, &fail, stream);
            if (e == cudaSuccess) {
                static std::atomic<bool> told{false};
                if (!told.exchange(true))
                    std::fprintf(stderr, "strata: uploads batched with cudaMemcpyBatchAsync (CUDA 12.8+ form, mode %d)\n", batch_mode);
                return cudaSuccess;
            }
            const bool for_good = e == cudaErrorCallRequiresNewerDriver || e == cudaErrorNotSupported ||
                                  e == cudaErrorNotYetImplemented;
            if (for_good) batch_off.store(true);
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "strata: cudaMemcpyBatchAsync failed (%s); the uploads go one copy at a time%s\n",
                             cudaGetErrorString(e), for_good ? " from now on" : "");
            }
            (void) cudaGetLastError();
        }
    }
#elif defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 7
    // QuantLab (11/16): ROCm 7+ has the batch with CUDA 12.8's signature (the failing index).  The sources go as they are:
    // on Windows the HIP runtime gives a registered pointer's "device pointer" as the same address, and it tells host
    // memory by the pointer itself.  A refused batch is redone as the loop (one warning).  Never run on an AMD card.
    constexpr size_t cap = 128;
    if (batch_mode != 0 && n > 1 && n <= cap) {
        hipMemcpyAttributes attr{};
        attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
        if (batch_mode == 2) attr.flags = hipMemcpyFlagPreferOverlapWithCompute;
        size_t first = 0, fail = 0;
        std::array<void*, cap> dsts{}, srcs{};
        std::array<size_t, cap> sizes{};
        for (size_t i = 0; i < n; ++i) { dsts[i] = dst[i]; srcs[i] = const_cast<void*>(src[i]); sizes[i] = bytes[i]; }
        const hipError_t e = hipMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), n, &attr, &first, 1, &fail, stream);
        if (e == hipSuccess) return cudaSuccess;
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "strata: hipMemcpyBatchAsync failed (%s); the uploads go one copy at a time\n", hipGetErrorString(e));
        }
        (void) hipGetLastError();
    }
#else
    (void) batch_mode;
#endif
    for (size_t i = 0; i < n; ++i) {
        const cudaError_t e = cudaMemcpyAsync(dst[i], src[i], bytes[i], cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

/// The verifier's form: `n` blobs of `bytes` each into consecutive slots of `dst`.
inline cudaError_t copy_expert_blobs(uint8_t* dst, const uint8_t* const* src, int n, size_t bytes,
                                     cudaStream_t stream, int batch_mode) {
    if (n <= 0) return cudaSuccess;
    constexpr size_t cap = 128;
    if (batch_mode == 0 || (size_t) n > cap) {   // the old loop, with no staging of its own
        for (int i = 0; i < n; ++i) {
            const cudaError_t e = cudaMemcpyAsync(dst + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }
    void* d[cap] = {};   // plain arrays: nvcc on Linux (13.0, g++) rejects std::array<T, cap> here ("template argument 2 is invalid")
    const void* s[cap] = {};
    size_t b[cap] = {};
    for (int i = 0; i < n; ++i) { d[i] = dst + (size_t) i * bytes; s[i] = src[i]; b[i] = bytes; }
    return copy_blobs(d, s, b, (size_t) n, stream, batch_mode);
}

}  // namespace strata::core
