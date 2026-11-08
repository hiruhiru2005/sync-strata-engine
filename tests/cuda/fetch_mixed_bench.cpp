// fetch_mixed_bench - QuantLab (11/9): the decode's PCIe share copy (strata::kernels::fetch_blobs_mixed) from MAPPED HOST
// memory, as the verifier stages a mixed layer's R1Q2 experts: the 10/31 kernel (impl 0) and the per-blob kernel with four
// loads in flight (impl 1).  Twelve blobs of mixed K (the model's sizes: n_embd 2048, n_ff 640), 50 copies each way after
// a warm-up: GB/s of blob bytes and the staging bytes compared byte for byte.
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("%s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}
}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("fetch_mixed_bench: no CUDA device, skipped\n");
        return 77;
    }
    const int64_t n_embd = 2048, n_ff = 640;
    const int64_t gu_unit = 2 * n_ff * (n_embd / 128), d_unit = n_embd * (n_ff / 128);
    const int64_t sc_bytes = 16000;
    auto own = [&](int kg, int kd) {
        return ((gu_unit * 4 * kg + d_unit * 4 * kd + 15) / 16 * 16 + sc_bytes + 255) / 256 * 256;
    };
    const int64_t slot = (own(14, 14) + 255) / 256 * 256;
    const int NE = 12;
    const int ks[NE][2] = {{9, 10}, {10, 11}, {11, 12}, {12, 13}, {13, 14}, {8, 9}, {9, 9}, {10, 10}, {11, 11}, {12, 12}, {14, 14}, {7, 8}};
    int64_t bytes = 0;
    for (int i = 0; i < NE; ++i) bytes += own(ks[i][0], ks[i][1]);
    // the blobs in one mapped (page-locked) host buffer, as the RAM copy's registered prefix
    uint8_t* host = nullptr;
    if (!ck(cudaHostAlloc((void**) &host, (size_t) slot * NE, cudaHostAllocMapped), "cudaHostAlloc")) return 1;
    std::mt19937 rng(1109);
    for (int64_t i = 0; i < slot * NE; ++i) host[i] = (uint8_t) (rng() & 0xFF);
    uint8_t* hdev = nullptr;
    if (!ck(cudaHostGetDevicePointer((void**) &hdev, host, 0), "cudaHostGetDevicePointer")) return 1;
    std::vector<unsigned long long> ptr(NE);
    for (int i = 0; i < NE; ++i)
        ptr[i] = (unsigned long long) (hdev + (size_t) i * slot) | ((unsigned long long) (ks[i][0] | (ks[i][1] << 4)) << 56);
    unsigned long long* d_ptr = nullptr;
    int32_t* d_n = nullptr;
    uint8_t *st0 = nullptr, *st1 = nullptr;
    if (!ck(cudaMalloc(&d_ptr, sizeof(unsigned long long) * 16), "malloc ptr") || !ck(cudaMalloc(&d_n, 4), "malloc n") ||
        !ck(cudaMalloc(&st0, (size_t) slot * 16), "malloc st0") || !ck(cudaMalloc(&st1, (size_t) slot * 16), "malloc st1"))
        return 1;
    const int32_t n = NE;
    if (!ck(cudaMemcpy(d_ptr, ptr.data(), sizeof(unsigned long long) * NE, cudaMemcpyHostToDevice), "up ptr") ||
        !ck(cudaMemcpy(d_n, &n, 4, cudaMemcpyHostToDevice), "up n") || !ck(cudaMemset(st0, 0, (size_t) slot * 16), "zero0") ||
        !ck(cudaMemset(st1, 0, (size_t) slot * 16), "zero1"))
        return 1;
    cudaStream_t s;
    cudaEvent_t e0, e1;
    cudaStreamCreate(&s);
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double gbs[2] = {};
    for (int impl = 0; impl < 2; ++impl) {
        uint8_t* st = impl ? st1 : st0;
        for (int w = 0; w < 5; ++w) strata::kernels::fetch_blobs_mixed_impl(d_ptr, d_n, st, slot, gu_unit, d_unit, 16, s, sc_bytes, impl);
        cudaEventRecord(e0, s);
        const int reps = 50;
        for (int r = 0; r < reps; ++r) strata::kernels::fetch_blobs_mixed_impl(d_ptr, d_n, st, slot, gu_unit, d_unit, 16, s, sc_bytes, impl);
        cudaEventRecord(e1, s);
        if (!ck(cudaEventSynchronize(e1), "sync")) return 1;
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        gbs[impl] = (double) bytes * reps / (ms / 1e3) / 1e9;
        std::printf("impl %d: %.3f ms a copy of %.2f MB (%d blobs): %.1f GB/s\n", impl, ms / reps, bytes / 1e6, NE, gbs[impl]);
    }
    std::vector<uint8_t> a((size_t) slot * 16), b((size_t) slot * 16);
    cudaMemcpy(a.data(), st0, a.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), st1, b.size(), cudaMemcpyDeviceToHost);
    bool same = a == b;
    bool src_ok = true;   // and each blob's own bytes really are the host's
    for (int i = 0; i < NE && src_ok; ++i)
        src_ok = std::memcmp(b.data() + (size_t) i * slot, host + (size_t) i * slot, (size_t) own(ks[i][0], ks[i][1])) == 0;
    std::printf("fetch_mixed_bench: staging %s, blob bytes %s, speed-up %.2fx - %s\n", same ? "same" : "DIFFERENT",
                src_ok ? "match the host" : "DIFFER", gbs[1] / std::max(gbs[0], 1e-9), same && src_ok ? "PASS" : "FAIL");
    cudaFreeHost(host);
    cudaFree(d_ptr);
    cudaFree(d_n);
    cudaFree(st0);
    cudaFree(st1);
    return same && src_ok ? 0 : 1;
}
