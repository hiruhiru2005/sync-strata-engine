// prefill_r1q2_batch_test - QuantLab (11/9): mmq::gather_r1q2_batch (R1Q2 experts of any K, one launch per 32) against
// one mmq::gather_r1q2 an expert and against a CPU reference of gather_r1q2's formula (each block's fp16 d =
// fp16(colscale[b] * p2[m][q[r][b]] * rowbase[r]) before its codes), word for word over the whole slots and the zeroed
// tails.  37 experts with mixed K (2..14) and the model's sizes (n_embd 2048, n_ff 640): two batch launches.
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr int64_t kEmbd = 2048, kFf = 640;
constexpr int kNbe = (int) (kEmbd / 128), kNbf = (int) (kFf / 128);
constexpr size_t kTail = 4096;   // MMQ_TAIL

struct Expert {
    int kg = 0, kd = 0;
    size_t up_off = 0, down_off = 0, sc_off = 0, bytes = 0;
    size_t gu_words = 0, d_words = 0;   // the expanded slots
    float p2[3][4] = {};
    std::vector<uint8_t> blob;
};

uint16_t half_bits(float v) {
    const __half h = __float2half(v);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
float half_float(uint16_t b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
}

Expert make_expert(int kg, int kd, std::mt19937& rng) {
    Expert e;
    e.kg = kg;
    e.kd = kd;
    e.up_off = (size_t) kFf * kNbe * 4 * kg;
    e.down_off = 2 * e.up_off;
    e.sc_off = e.down_off + (size_t) kEmbd * kNbf * 4 * kd;
    const size_t qge = (kNbe + 3) / 4, qdf = (kNbf + 3) / 4;
    const size_t g_sc = kFf * 2 + kNbe * 2 + kFf * qge, d_sc = kEmbd * 2 + kNbf * 2 + kEmbd * qdf;
    e.bytes = e.sc_off + 2 * g_sc + d_sc;
    e.blob.resize(e.bytes);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.25f, 4.0f), sgn(-1.0f, 1.0f);
    for (size_t i = 0; i < e.sc_off; ++i) e.blob[i] = (uint8_t) byte(rng);   // the codes
    for (int m = 0; m < 3; ++m) {   // per matrix: rowbase fp16[rows], colscale fp16[n_in / 128], q 2-bit [rows][qb]
        const size_t rows = m == 2 ? kEmbd : kFf, nb = m == 2 ? kNbf : kNbe, qb = m == 2 ? qdf : qge;
        uint8_t* s = e.blob.data() + e.sc_off + (size_t) m * g_sc;
        for (size_t r = 0; r < rows; ++r) { const uint16_t h = half_bits(scale(rng)); std::memcpy(s + 2 * r, &h, 2); }
        for (size_t b = 0; b < nb; ++b) { const uint16_t h = half_bits(scale(rng) * 1e-3f); std::memcpy(s + 2 * rows + 2 * b, &h, 2); }
        for (size_t i = 0; i < rows * qb; ++i) s[2 * rows + 2 * nb + i] = (uint8_t) byte(rng);
        for (int q = 0; q < 4; ++q) e.p2[m][q] = sgn(rng) * 3.0f;
    }
    e.gu_words = (size_t) 2 * kFf * kNbe * (1 + 2 * kg);
    e.d_words = (size_t) kEmbd * kNbf * (1 + 2 * kd);
    return e;
}

// gather_r1q2's formula on the CPU (the 10/31 kernel, word by word)
void reference(const Expert& e, std::vector<uint16_t>& gu, std::vector<uint16_t>& d) {
    const size_t qge = (kNbe + 3) / 4, qdf = (kNbf + 3) / 4;
    const size_t g_sc = kFf * 2 + kNbe * 2 + kFf * qge;
    const uint8_t* blob = e.blob.data();
    auto one = [&](bool down, size_t blk, uint16_t* dst) {
        const int K = down ? e.kd : e.kg;
        const uint16_t* src = (const uint16_t*) (down ? blob + e.down_off : blob);
        size_t r, rows, nb, qb, base;
        int m, b;
        if (down) {
            m = 2; nb = kNbf; qb = qdf; rows = kEmbd; r = blk / kNbf; b = (int) (blk % kNbf); base = e.sc_off + 2 * g_sc;
        } else {
            m = blk >= (size_t) kFf * kNbe ? 1 : 0;
            const size_t bb = blk - (size_t) m * kFf * kNbe;
            nb = kNbe; qb = qge; rows = kFf; r = bb / kNbe; b = (int) (bb % kNbe); base = e.sc_off + (size_t) m * g_sc;
        }
        const uint8_t* sc = blob + base;
        uint16_t rbh, csh;
        std::memcpy(&rbh, sc + 2 * r, 2);
        std::memcpy(&csh, sc + 2 * rows + 2 * (size_t) b, 2);
        const int q = (sc[rows * 2 + nb * 2 + r * qb + (size_t) (b >> 2)] >> (2 * (b & 3))) & 3;
        const float cs = half_float(csh), pq = e.p2[m][q], rb = half_float(rbh);
        dst[0] = half_bits(cs * pq * rb);
        for (int j = 0; j < 2 * K; ++j) dst[1 + j] = src[blk * 2 * K + (size_t) j];
    };
    for (size_t blk = 0; blk < (size_t) 2 * kFf * kNbe; ++blk) one(false, blk, gu.data() + blk * (1 + 2 * e.kg));
    for (size_t blk = 0; blk < (size_t) kEmbd * kNbf; ++blk) one(true, blk, d.data() + blk * (1 + 2 * e.kd));
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("prefill_r1q2_batch_test: no CUDA device, skipped\n");
        return 77;
    }
    try {
        std::mt19937 rng(1109);
        const int ks[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        std::vector<Expert> ex;
        for (int i = 0; i < 37; ++i) ex.push_back(make_expert(ks[(i * 5) % 13], ks[(i * 7 + 3) % 13], rng));
        int fails = 0;
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        std::vector<void*> blobs, gu1, d1, gu2, d2;
        std::vector<mmq::R1q2Item> items;
        for (size_t i = 0; i < ex.size(); ++i) {
            const Expert& e = ex[i];
            void *b, *a1, *a2, *c1, *c2;
            const size_t gub = e.gu_words * 2 + kTail, db = e.d_words * 2 + kTail;
            ck(cudaMalloc(&b, e.bytes), "blob");
            ck(cudaMemcpy(b, e.blob.data(), e.bytes, cudaMemcpyHostToDevice), "blob up");
            ck(cudaMalloc(&a1, gub), "gu1");
            ck(cudaMalloc(&a2, gub), "gu2");
            ck(cudaMalloc(&c1, db), "d1");
            ck(cudaMalloc(&c2, db), "d2");
            for (void* p : {a1, a2}) ck(cudaMemset(p, 0xAB, gub), "fill");   // the tails must come out zeroed
            for (void* p : {c1, c2}) ck(cudaMemset(p, 0xAB, db), "fill");
            blobs.push_back(b); gu1.push_back(a1); gu2.push_back(a2); d1.push_back(c1); d2.push_back(c2);
            const size_t zero = i % 3 == 0 ? kTail : 0;
            mmq::gather_r1q2((const uint8_t*) b, e.kg, e.kd, kEmbd, kFf, e.up_off, e.down_off, e.sc_off, e.p2, a1, c1, s, zero);
            mmq::R1q2Item it;
            it.blob = (const uint8_t*) b;
            it.kg = e.kg;
            it.kd = e.kd;
            it.up_off = e.up_off;
            it.down_off = e.down_off;
            it.sc_off = e.sc_off;
            it.p2 = e.p2;
            it.gu_dst = a2;
            it.d_dst = c2;
            it.zero_bytes = zero;
            items.push_back(it);
        }
        mmq::gather_r1q2_batch(items.data(), (int) items.size(), kEmbd, kFf, s);
        ck(cudaStreamSynchronize(s), "sync");
        for (size_t i = 0; i < ex.size(); ++i) {
            const Expert& e = ex[i];
            const size_t gub = e.gu_words * 2 + kTail, db = e.d_words * 2 + kTail;
            std::vector<uint8_t> h1(gub), h2(gub), k1(db), k2(db);
            ck(cudaMemcpy(h1.data(), gu1[i], gub, cudaMemcpyDeviceToHost), "down gu1");
            ck(cudaMemcpy(h2.data(), gu2[i], gub, cudaMemcpyDeviceToHost), "down gu2");
            ck(cudaMemcpy(k1.data(), d1[i], db, cudaMemcpyDeviceToHost), "down d1");
            ck(cudaMemcpy(k2.data(), d2[i], db, cudaMemcpyDeviceToHost), "down d2");
            std::vector<uint16_t> rg(e.gu_words), rd(e.d_words);
            reference(e, rg, rd);
            std::vector<uint8_t> want_g(gub, 0xAB), want_d(db, 0xAB);
            std::memcpy(want_g.data(), rg.data(), e.gu_words * 2);
            std::memcpy(want_d.data(), rd.data(), e.d_words * 2);
            if (i % 3 == 0) {
                std::memset(want_g.data() + e.gu_words * 2, 0, kTail);
                std::memset(want_d.data() + e.d_words * 2, 0, kTail);
            }
            const bool one_ok = h1 == want_g && k1 == want_d, batch_ok = h2 == want_g && k2 == want_d;
            if (!one_ok || !batch_ok) {
                ++fails;
                std::printf("  expert %zu (K %d / %d): one-by-one %s, batch %s\n", i, e.kg, e.kd, one_ok ? "ok" : "DIFFERS",
                            batch_ok ? "ok" : "DIFFERS");
            }
        }
        for (size_t i = 0; i < ex.size(); ++i) {
            cudaFree(blobs[i]); cudaFree(gu1[i]); cudaFree(gu2[i]); cudaFree(d1[i]); cudaFree(d2[i]);
        }
        cudaStreamDestroy(s);
        std::printf("prefill_r1q2_batch_test: %zu experts, %d differ - %s\n", ex.size(), fails, fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("prefill_r1q2_batch_test: %s\n", e.what());
        return 1;
    }
}
