// src/core/hip_arena.cu - QuantLab (11/19) STRATA_HIP_ARENA (strata/core/hip_arena.hpp): how the RAM experts' page-locked
// copy is taken on an AMD card.  Added to strata_engine by HIP builds only (CMakeLists.txt): a CUDA build neither compiles
// nor links this file, and every caller sits under #if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906).
//
// Why: an RX 9070 XT (Windows) spent 42-64 ms of a 63-86 ms decode window waiting for the PCIe share, and the copy the
// GPU reads it from was fine-grained (cudaHostAlloc(Mapped | Portable): HIP makes a mapped block coherent), which the GPU
// does not cache.  Coarse-grained host memory may be cached by the GPU and is coherent only at dispatch boundaries (the HIP
// header's own words for hipMemAdviseSetCoarseGrain).  That is enough for this copy:
//  - the GPU only reads it (no GPU write ever has to reach the host);
//  - the host rewrites a place of it only while no published plan names the place (commit_exchanges / commit_copies,
//    install_resident, ql_take_into_ram, ql_swap_pin, the RAM_ELASTIC refill: the place has left the table, or the window
//    that may still read it has been fenced), and the plans that name the new bytes are published after the write; the
//    kernel that reads them (fetch_blobs / the PCIe grouped launch, verify.cpp) is a dispatch that starts after the wait
//    for that plan's flag, so after the write.
// Whether this card and runtime really refresh a coarse-grained block at every dispatch boundary - inside one stream with
// no host synchronisation, and inside one graph, as the decode runs - is not taken on trust: the stale-read test below
// does exactly that sequence (read the block, rewrite it from the host while a later kernel of the same stream waits on
// a flag, read it again) and a way that fails it, or whose test does not run cleanly, is never picked by auto.
//
// STRATA_HIP_ARENA=auto (default) | coherent (as before, nothing measured) | noncoherent | register (forced: a refused
// allocation still falls back to coherent).  STRATA_HIP_ARENA_PROBE_MIB: the measured region (default 64, 16..512).
#include "strata/core/hip_arena.hpp"

#include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <stdlib.h>
#endif

namespace strata::core::hip_arena {
namespace {

// hipHostRegister's coarse-grained flag (ROCm 6+ headers; the same bit as hipHostRegisterReadOnly, which AMD ignores)
#ifdef hipExtHostRegisterCoarseGrained
constexpr unsigned kRegCoarse = hipExtHostRegisterCoarseGrained;
#else
constexpr unsigned kRegCoarse = 0x8u;
#endif

// wall_clock64's rate, as verify_kernels.cu's strata_now_ns has it (gfx906: 40 ns a tick, RDNA: 10 ns)
#if defined(STRATA_HIP_GFX906)
constexpr unsigned long long kTicksPerMs = 25000ull;
#else
constexpr unsigned long long kTicksPerMs = 100000ull;
#endif
constexpr unsigned long long kWaitTicks = 300ull * kTicksPerMs;   // the test's GPU wait gives up after 300 ms (TDR is 2 s)

constexpr int kThreads = 256;
constexpr size_t kStaleBytes = 256u << 10;   // under every supported card's L2 (2 MiB on the smallest RDNA3 / iGPU)
constexpr double kGiB = 1073741824.0;

// ---- device side ----------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned ha_block_xor(unsigned acc) {
    __shared__ unsigned s[kThreads];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned w = kThreads / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) s[threadIdx.x] ^= s[threadIdx.x + w];
        __syncthreads();
    }
    return s[0];
}

// every 16 bytes of p[0, n16) read once and XOR-folded (the fold keeps the loads, and is the stale test's checksum)
__global__ void ha_read_kernel(const uint4* __restrict__ p, unsigned long long n16, unsigned* __restrict__ out) {
    unsigned acc = 0;
    const unsigned long long stride = (unsigned long long) gridDim.x * kThreads;
    for (unsigned long long i = (unsigned long long) blockIdx.x * kThreads + threadIdx.x; i < n16; i += stride) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    const unsigned b = ha_block_xor(acc);
    if (threadIdx.x == 0) atomicXor(out, b);
}

// the stale test's first read (one block): the region through the GPU's caches, then *done = token for the host
__global__ void ha_read_signal_kernel(const uint4* __restrict__ p, unsigned long long n16, unsigned* __restrict__ out,
                                      volatile unsigned* done, unsigned token) {
    unsigned acc = 0;
    for (unsigned long long i = threadIdx.x; i < n16; i += kThreads) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    const unsigned b = ha_block_xor(acc);
    if (threadIdx.x == 0) {
        *out = b;
        __threadfence_system();
        *done = token;
        __threadfence_system();
    }
}

// spins until the host sets *go = token (it rewrote the region meanwhile), bounded: *status = 1 seen, 2 gave up
__global__ void ha_wait_kernel(const volatile unsigned* go, unsigned token, unsigned* status, unsigned long long limit) {
    const unsigned long long t0 = (unsigned long long) wall_clock64();
    unsigned st = 1;
    for (unsigned long long n = 0; *go != token; ++n) {
        if ((n & 255ull) == 0 && (unsigned long long) wall_clock64() - t0 > limit) { st = 2; break; }
        if (n > (1ull << 32)) { st = 2; break; }   // a clock that does not move
        __builtin_amdgcn_s_sleep(2);
    }
    __threadfence_system();
    *status = st;
}

// ---- host side ------------------------------------------------------------------------------------------------------
struct Region {
    void* host = nullptr;
    const void* dev = nullptr;
    int mode = kCoherent;
    bool ok = false;
    std::string why;
};

std::mutex g_reg_mu;
std::unordered_map<void*, size_t> g_registered;   // alloc's kRegister blocks: free_arena unregisters and releases them

void* page_alloc(size_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, 4096, bytes) == 0 ? p : nullptr;
#endif
}

void page_free(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    (void) VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

void drop(Region& r) {
    if (r.host != nullptr) {
        if (r.mode == kRegister) {
            (void) hipHostUnregister(r.host);
            (void) hipGetLastError();
            page_free(r.host);
        } else {
            (void) cudaFreeHost(r.host);
            (void) hipGetLastError();
        }
    }
    r.host = nullptr;
    r.dev = nullptr;
    r.ok = false;
}

// a block of `bytes` taken `mode`'s way, its device alias checked (r.ok); r.why says why not
Region take(int mode, size_t bytes) {
    Region r;
    r.mode = mode;
    hipError_t e = hipSuccess;
    if (mode == kCoherent) {
        e = cudaHostAlloc(&r.host, bytes, cudaHostAllocMapped | cudaHostAllocPortable);   // the allocation as before
    } else if (mode == kNonCoherent) {
        // not Mapped: HIP makes a Mapped block fine-grained whatever else is asked; an AMD host block has its alias anyway
        e = cudaHostAlloc(&r.host, bytes, cudaHostAllocPortable | hipHostMallocNonCoherent);
    } else {
        r.host = page_alloc(bytes);
        if (r.host == nullptr) {
            r.why = "no memory for the block";
            return r;
        }
        e = cudaHostRegister(r.host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable | kRegCoarse);
        if (e != hipSuccess) {
            page_free(r.host);
            r.host = nullptr;
        }
    }
    if (e != hipSuccess || r.host == nullptr) {
        r.why = e != hipSuccess ? hipGetErrorName(e) : "no block";
        (void) hipGetLastError();
        r.host = nullptr;
        return r;
    }
    void* d = nullptr;
    const hipError_t a = hipHostGetDevicePointer(&d, r.host, 0);
    if (a != hipSuccess || d == nullptr) {
        r.why = std::string("no device alias (") + hipGetErrorName(a) + ")";
        (void) hipGetLastError();
        drop(r);
        return r;
    }
    r.dev = d;
    r.ok = true;
    return r;
}

struct Ctx {
    hipStream_t st = nullptr;
    unsigned* d_words = nullptr;   // [0] first read, [1] second read, [2] wait status, [3] bandwidth sink
    void* d_buf = nullptr;         // the copy engine's destination (0: not measured)
    size_t buf_bytes = 0;
    void* h_flag_block = nullptr;  // coherent mapped: [0] done, [1] go
    volatile unsigned* h_flags = nullptr;
    unsigned* d_flags = nullptr;
    int sms = 1;
    unsigned token = 0x51A70000u;
};

struct Meas {
    bool ok = false;
    std::string why;
    double gpu = 0, copy = 0, cpu = 0;   // GB/s (0: not measured)
    int stale = -1;                      // -1 not run, 0 fresh every round, 1 a stale read seen, 2 inconclusive
    bool graph = false;                  // the graph rounds ran
};

volatile uint64_t g_sink = 0;

uint32_t pattern(uint32_t i, uint32_t seed) {
    uint32_t x = i * 0x9E3779B1u ^ seed;
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    return x;
}

uint32_t fill(void* p, size_t bytes, uint32_t seed) {
    uint32_t* w = (uint32_t*) p;
    uint32_t x = 0;
    for (size_t i = 0; i < bytes / 4; ++i) {
        const uint32_t v = pattern((uint32_t) i, seed);
        w[i] = v;
        x ^= v;
    }
    return x;
}

double cpu_read_gbs(const void* p, size_t bytes) {   // the CPU experts read the copy too: a block the CPU cannot cache loses
    const uint64_t* w = (const uint64_t*) p;
    double best = 0;
    for (int rep = 0; rep < 2; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t s = 0;
        for (size_t i = 0; i < bytes / 8; ++i) s += w[i];
        g_sink = s;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (sec > 0) best = std::max(best, (double) bytes / sec / 1e9);
    }
    return best;
}

double gpu_read_gbs(const void* dev, size_t bytes, Ctx& c) {   // --pcie-mode kernel / direct: a kernel reads the alias
    const unsigned long long n16 = bytes / 16;
    const unsigned long long want = (unsigned long long) std::max(1, c.sms) * 8ull;
    const int blocks = (int) std::max(1ull, std::min(want, (n16 + kThreads - 1) / kThreads));
    hipEvent_t e0 = nullptr, e1 = nullptr;
    if (hipEventCreate(&e0) != hipSuccess || hipEventCreate(&e1) != hipSuccess) {
        (void) hipGetLastError();
        if (e0 != nullptr) (void) hipEventDestroy(e0);
        return 0;
    }
    float best = 0;
    for (int rep = 0; rep < 4; ++rep) {   // the first one warms up
        (void) hipEventRecord(e0, c.st);
        ha_read_kernel<<<blocks, kThreads, 0, c.st>>>((const uint4*) dev, n16, c.d_words + 3);
        (void) hipEventRecord(e1, c.st);
        float ms = 0;
        if (hipEventSynchronize(e1) != hipSuccess || hipEventElapsedTime(&ms, e0, e1) != hipSuccess) {
            (void) hipGetLastError();
            best = 0;
            break;
        }
        if (rep > 0 && ms > 0 && (best == 0 || ms < best)) best = ms;
    }
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
    return best > 0 ? (double) bytes / ((double) best * 1e6) : 0;
}

double copy_gbs(const void* host, size_t bytes, Ctx& c) {   // --pcie-mode dma: the copy engine reads the block
    if (c.d_buf == nullptr || c.buf_bytes == 0) return 0;
#if defined(_WIN32)
    // QuantLab (11/20, 0.1.3): Windows HIP: the events do not bracket the copies there (an RX 7800 XT read 1,382 GB/s; the
    // PCIe probe in generate.cpp found the same on an RX 6800), so each pass is timed on the host between two synchronizes
    // of the stream.  The region (64 MiB by default) takes ~2.5 ms on an x16 PCIe 4 link, so the synchronize hardly counts,
    // and it can only make a reading slower, never faster than the link.
    if (hipStreamSynchronize(c.st) != hipSuccess) {
        (void) hipGetLastError();
        return 0;
    }
    double best_ms = 0;
    for (int rep = 0; rep < 3; ++rep) {   // the first one warms up
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = true;
        for (size_t off = 0; off < bytes && ok; off += c.buf_bytes)
            ok = hipMemcpyAsync(c.d_buf, (const uint8_t*) host + off, std::min(c.buf_bytes, bytes - off), hipMemcpyHostToDevice,
                                c.st) == hipSuccess;
        if (!ok || hipStreamSynchronize(c.st) != hipSuccess) {
            (void) hipGetLastError();
            (void) hipStreamSynchronize(c.st);
            best_ms = 0;
            break;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rep > 0 && ms > 0 && (best_ms == 0 || ms < best_ms)) best_ms = ms;
    }
    return best_ms > 0 ? (double) bytes / (best_ms * 1e6) : 0;
#else
    hipEvent_t e0 = nullptr, e1 = nullptr;
    if (hipEventCreate(&e0) != hipSuccess || hipEventCreate(&e1) != hipSuccess) {
        (void) hipGetLastError();
        if (e0 != nullptr) (void) hipEventDestroy(e0);
        return 0;
    }
    float best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        (void) hipEventRecord(e0, c.st);
        bool ok = true;
        for (size_t off = 0; off < bytes && ok; off += c.buf_bytes)
            ok = hipMemcpyAsync(c.d_buf, (const uint8_t*) host + off, std::min(c.buf_bytes, bytes - off), hipMemcpyHostToDevice,
                                c.st) == hipSuccess;
        (void) hipEventRecord(e1, c.st);
        float ms = 0;
        if (!ok || hipEventSynchronize(e1) != hipSuccess || hipEventElapsedTime(&ms, e0, e1) != hipSuccess) {
            (void) hipGetLastError();
            (void) hipStreamSynchronize(c.st);
            best = 0;
            break;
        }
        if (rep > 0 && ms > 0 && (best == 0 || ms < best)) best = ms;
    }
    (void) hipEventDestroy(e0);
    (void) hipEventDestroy(e1);
    return best > 0 ? (double) bytes / ((double) best * 1e6) : 0;
#endif
}

// The decode's sequence on the block's first kStaleBytes, twice: a kernel reads it (and says so on a coherent flag), the
// next kernel of the same stream waits for the host's go, the host rewrites the region and says go, the kernel after
// reads it again.  No host synchronisation between the two reads, as in a window; `graph`: the three kernels are one
// graph, as a captured window is.  0 fresh both times, 1 a stale read, 2 the test did not run as meant, 3 no graph here.
int stale_rounds(const Region& r, Ctx& c, bool graph, std::string& why) {
    const unsigned long long n16 = kStaleBytes / 16;
    for (int round = 0; round < 2; ++round) {
        const unsigned t = ++c.token;
        const uint32_t xa = fill(r.host, kStaleBytes, 0xA5000000u ^ t);
        c.h_flags[0] = 0u;
        c.h_flags[1] = 0u;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (hipMemsetAsync(c.d_words, 0, 3 * sizeof(unsigned), c.st) != hipSuccess) {
            (void) hipGetLastError();
            why = "memset refused";
            return 2;
        }
        hipGraph_t g = nullptr;
        hipGraphExec_t ge = nullptr;
        auto enqueue = [&]() {
            ha_read_signal_kernel<<<1, kThreads, 0, c.st>>>((const uint4*) r.dev, n16, c.d_words + 0, c.d_flags + 0, t);
            ha_wait_kernel<<<1, 1, 0, c.st>>>(c.d_flags + 1, t, c.d_words + 2, kWaitTicks);
            ha_read_kernel<<<16, kThreads, 0, c.st>>>((const uint4*) r.dev, n16, c.d_words + 1);
        };
        if (graph) {
            if (hipStreamBeginCapture(c.st, hipStreamCaptureModeThreadLocal) != hipSuccess) {
                (void) hipGetLastError();
                why = "no stream capture";
                return 3;
            }
            enqueue();
            const bool built = hipStreamEndCapture(c.st, &g) == hipSuccess && g != nullptr &&
                               hipGraphInstantiate(&ge, g, nullptr, nullptr, 0) == hipSuccess;
            if (!built || hipGraphLaunch(ge, c.st) != hipSuccess) {
                (void) hipGetLastError();
                if (ge != nullptr) (void) hipGraphExecDestroy(ge);
                if (g != nullptr) (void) hipGraphDestroy(g);
                (void) hipStreamSynchronize(c.st);
                (void) hipGetLastError();
                why = "no graph";
                return 3;
            }
        } else {
            enqueue();
        }
        // the first read has finished (the region now sits in the GPU's caches, if they keep it)
        const auto t0 = std::chrono::steady_clock::now();
        bool seen = true;
        while (c.h_flags[0] != t) {
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
                seen = false;
                break;
            }
            std::this_thread::yield();
        }
        const uint32_t xb = fill(r.host, kStaleBytes, 0x5A000000u ^ t);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        c.h_flags[1] = t;   // (set even when the first read was not seen: the wait ends either way)
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const hipError_t se = hipStreamSynchronize(c.st);
        if (ge != nullptr) (void) hipGraphExecDestroy(ge);
        if (g != nullptr) (void) hipGraphDestroy(g);
        unsigned w[3] = {0u, 0u, 0u};
        if (se != hipSuccess || hipMemcpy(w, c.d_words, sizeof w, hipMemcpyDeviceToHost) != hipSuccess) {
            (void) hipGetLastError();
            why = "the test's kernels failed";
            return 2;
        }
        if (!seen || w[2] != 1u || w[0] != xa) {
            why = !seen ? "the first read was not seen" : w[2] != 1u ? "the wait gave up" : "the first read was wrong";
            return 2;
        }
        if (w[1] != xb) {
            why = w[1] == xa ? "the second read saw the old bytes" : "the second read saw old and new bytes mixed";
            return 1;
        }
    }
    return 0;
}

Meas measure(int mode, size_t bytes, Ctx& c) {
    Meas m;
    Region r = take(mode, bytes);
    if (!r.ok) {
        m.why = r.why;
        return m;
    }
    std::memset(r.host, 0x5a, bytes);
    m.cpu = cpu_read_gbs(r.host, std::min<size_t>(bytes, 16u << 20));
    m.gpu = gpu_read_gbs(r.dev, bytes, c);
    m.copy = copy_gbs(r.host, bytes, c);
    std::string why;
    m.stale = stale_rounds(r, c, false, why);
    if (m.stale == 0) {
        std::string gwhy;
        const int gs = stale_rounds(r, c, true, gwhy);
        m.graph = gs != 3;
        if (gs == 1 || gs == 2) {
            m.stale = gs;
            why = "graph: " + gwhy;
        }
    }
    if (m.stale != 0) m.why = why;
    drop(r);
    m.ok = true;
    return m;
}

int g_cached = -1;
std::mutex g_choose_mu;

}  // namespace

const char* mode_name(int mode) {
    return mode == kNonCoherent ? "noncoherent" : mode == kRegister ? "register" : "coherent";
}

int choose(size_t bytes) {
    std::lock_guard<std::mutex> lk(g_choose_mu);
    if (g_cached >= 0) return g_cached;
    std::string v;
    if (const char* e = std::getenv("STRATA_HIP_ARENA")) v = e;
    for (char& ch : v) ch = (char) std::tolower((unsigned char) ch);
    if (v == "coherent" || v == "0" || v == "off") {
        std::fprintf(stderr, "FileExpertSource: QuantLab: STRATA_HIP_ARENA=coherent: the RAM copy is taken coherent "
                             "(fine-grained) as before, nothing measured\n");
        return g_cached = kCoherent;
    }
    int forced = -1;
    if (v == "noncoherent" || v == "nc") forced = kNonCoherent;
    else if (v == "register" || v == "reg") forced = kRegister;
    else if (!v.empty() && v != "auto" && v != "1")
        std::fprintf(stderr, "FileExpertSource: QuantLab: STRATA_HIP_ARENA=%s is not auto|coherent|noncoherent|register: "
                             "auto\n", v.c_str());
    size_t probe = 64u << 20;
    if (const char* e = std::getenv("STRATA_HIP_ARENA_PROBE_MIB")) {
        const int mib = std::atoi(e);
        if (mib > 0) probe = (size_t) std::max(16, std::min(512, mib)) << 20;
    }
    if (bytes > 0) probe = std::min(probe, std::max<size_t>(bytes, kStaleBytes) / 4096 * 4096);   // a small copy: no larger
    probe = std::max<size_t>(probe, kStaleBytes);

    const auto t0 = std::chrono::steady_clock::now();
    Ctx c;
    Meas m[3];
    std::string setup;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetAttribute(&c.sms, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || c.sms <= 0) {
        (void) hipGetLastError();
        c.sms = 16;
    }
    if (hipStreamCreateWithFlags(&c.st, hipStreamNonBlocking) != hipSuccess) {
        c.st = nullptr;
        setup = "no stream";
    } else if (hipMalloc((void**) &c.d_words, 4 * sizeof(unsigned)) != hipSuccess) {
        c.d_words = nullptr;
        setup = "no device words";
    } else if (cudaHostAlloc(&c.h_flag_block, 64, cudaHostAllocMapped | cudaHostAllocPortable | hipHostMallocCoherent) !=
                   hipSuccess ||
               hipHostGetDevicePointer((void**) &c.d_flags, c.h_flag_block, 0) != hipSuccess || c.d_flags == nullptr) {
        setup = "no coherent flags";
    } else {
        c.h_flags = (volatile unsigned*) c.h_flag_block;
        c.buf_bytes = 8u << 20;
        if (hipMalloc(&c.d_buf, c.buf_bytes) != hipSuccess) {   // the copy engine's figure is left out without it
            c.d_buf = nullptr;
            c.buf_bytes = 0;
        }
    }
    (void) hipGetLastError();
    if (setup.empty())
        for (int k = 0; k < 3; ++k) m[k] = measure(k, probe, c);
    if (c.d_buf != nullptr) (void) hipFree(c.d_buf);
    if (c.h_flag_block != nullptr) (void) cudaFreeHost(c.h_flag_block);
    if (c.d_words != nullptr) (void) hipFree(c.d_words);
    if (c.st != nullptr) (void) hipStreamDestroy(c.st);
    (void) hipGetLastError();

    // the rule: a coarse-grained way only when the coherent control measured cleanly (its stale test fresh: the test
    // works), the way's own stale test was fresh in the stream AND the graph rounds (where the graph could be built), its
    // GPU read beats coherent's by 5% (register: 10%, and the non-coherent block's by 5% - registering 13 GiB is the
    // heavier way), and neither the CPU's read (the CPU experts) nor the copy engine's (--pcie-mode dma) loses much
    int pick = kCoherent;
    std::string reason;
    const Meas& co = m[kCoherent];
    if (forced >= 0) {
        pick = forced;
        reason = std::string("forced by STRATA_HIP_ARENA") +
                 (m[forced].ok && m[forced].stale == 0 ? "" : "; WARNING: it did not pass the probe");
    } else if (!setup.empty()) {
        reason = "probe not run (" + setup + ")";
    } else if (!co.ok || co.gpu <= 0 || co.stale != 0) {
        reason = "the coherent control did not measure cleanly";
    } else {
        double need = co.gpu * 1.05;
        for (int k : {(int) kNonCoherent, (int) kRegister}) {
            const Meas& x = m[k];
            if (!x.ok || x.stale != 0 || x.gpu <= 0) continue;
            if (co.cpu > 0 && x.cpu < 0.8 * co.cpu) continue;
            if (co.copy > 0 && x.copy > 0 && x.copy < 0.9 * co.copy) continue;
            const double bar = k == kRegister ? std::max(need, co.gpu * 1.10) : need;
            if (x.gpu > bar) {
                pick = k;
                need = x.gpu * 1.05;
            }
        }
        reason = pick == kCoherent ? "no coarse-grained way was both safe here and faster" : "faster, stale test fresh";
    }
    std::string line;
    for (int k = 0; k < 3; ++k) {
        char b[256];
        if (!m[k].ok)
            std::snprintf(b, sizeof b, "%s%s %s", k ? "; " : "", mode_name(k),
                          setup.empty() ? ("not taken (" + m[k].why + ")").c_str() : "not measured");
        else
            std::snprintf(b, sizeof b, "%s%s GPU %.1f / copy %.1f / CPU %.1f GB/s, stale test %s%s%s%s", k ? "; " : "",
                          mode_name(k), m[k].gpu, m[k].copy, m[k].cpu,
                          m[k].stale == 0 ? "fresh" : m[k].stale == 1 ? "STALE" : "inconclusive",
                          m[k].stale == 0 ? (m[k].graph ? " (stream+graph)" : " (stream only)") : " (",
                          m[k].stale == 0 ? "" : m[k].why.c_str(), m[k].stale == 0 ? "" : ")");
        line += b;
    }
    std::fprintf(stderr, "FileExpertSource: QuantLab: STRATA_HIP_ARENA=%s: %.0f MiB probe in %.0f ms: %s -> %s (%s)\n",
                 forced >= 0 ? mode_name(forced) : "auto", (double) probe / 1048576.0,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), line.c_str(),
                 mode_name(pick), reason.c_str());
    std::fflush(stderr);
    return g_cached = pick;
}

hipError_t alloc(void** p, size_t bytes, int mode, int* taken) {
    *p = nullptr;
    if (taken != nullptr) *taken = kCoherent;
    if (mode == kNonCoherent || mode == kRegister) {
        Region r = take(mode, bytes);
        if (r.ok) {
            if (mode == kRegister) {
                std::lock_guard<std::mutex> lk(g_reg_mu);
                g_registered[r.host] = bytes;
            }
            *p = r.host;
            if (taken != nullptr) *taken = mode;
            return hipSuccess;
        }
        std::fprintf(stderr, "FileExpertSource: QuantLab: STRATA_HIP_ARENA: the %s copy of %.2f GiB was refused (%s); taking "
                             "it coherent as before\n", mode_name(mode), (double) bytes / kGiB, r.why.c_str());
    }
    return cudaHostAlloc(p, bytes, cudaHostAllocMapped | cudaHostAllocPortable);   // the allocation as before
}

void free_arena(void* p) {
    if (p == nullptr) return;
    {
        std::lock_guard<std::mutex> lk(g_reg_mu);
        const auto it = g_registered.find(p);
        if (it != g_registered.end()) {
            g_registered.erase(it);
            (void) hipHostUnregister(p);
            (void) hipGetLastError();
            page_free(p);
            return;
        }
    }
    (void) cudaFreeHost(p);
}

}  // namespace strata::core::hip_arena
