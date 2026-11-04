// src/kernels/tq_lut.cpp - QuantLab (11/4): the trellis codebook the engine decodes with; see strata/kernels/tq_lut.hpp.
#include "strata/kernels/tq_lut.hpp"

#include "ggml-common-tq.h"   // tq_lut_f16: the built-in codebook (host data in this translation unit)

// the built-in int8 copy, host data here too: the generator below must reproduce it from the built-in codebook
#ifndef __device__
#define __device__
#define STRATA_TQ_LUT_UNDEF_DEVICE
#endif
#include "cuda/tq_lut_i8.cuh"
#ifdef STRATA_TQ_LUT_UNDEF_DEVICE
#undef __device__
#undef STRATA_TQ_LUT_UNDEF_DEVICE
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {

static_assert(sizeof(tq_lut_f16) == kTqLutBytes, "the built-in codebook is 4096 fp16");
static_assert(sizeof(tq_lut_i8) == kTqLutPoints * 2, "the built-in int8 copy is 2048 pairs");

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {   // subnormal: normalize
            int e = -1;
            uint32_t m = man;
            do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 112u) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// ---- SHA-256 (FIPS 180-4), enough for a few small files
struct Sha256 {
    uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    uint8_t buf[64];
    size_t n = 0;
    uint64_t len = 0;
    static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
    void block(const uint8_t* p) {
        static const uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t) p[4 * i] << 24) | ((uint32_t) p[4 * i + 1] << 16) | ((uint32_t) p[4 * i + 2] << 8) | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void add(const void* data, size_t size) {
        const uint8_t* p = (const uint8_t*) data;
        len += size;
        while (size > 0) {
            const size_t take = std::min(size, (size_t) 64 - n);
            std::memcpy(buf + n, p, take);
            n += take; p += take; size -= take;
            if (n == 64) { block(buf); n = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = len * 8;
        const uint8_t one = 0x80, zero = 0;
        add(&one, 1);
        while (n != 56) add(&zero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; ++i) l[i] = (uint8_t) (bits >> (56 - 8 * i));
        add(l, 8);
        static const char* dg = "0123456789abcdef";
        std::string s;
        for (uint32_t x : h)
            for (int i = 28; i >= 0; i -= 4) s.push_back(dg[(x >> i) & 0xF]);
        return s;
    }
};

std::string sha256_hex(const void* data, size_t size) {
    Sha256 s;
    s.add(data, size);
    return s.hex();
}

struct State {
    alignas(16) uint16_t lut[kTqLutValues];
    bool builtin = true;
    std::string sha;
};

State& state() {
    static State s = [] {
        State t;
        std::memcpy(t.lut, tq_lut_f16, kTqLutBytes);
        t.sha = sha256_hex(t.lut, kTqLutBytes);
        return t;
    }();
    return s;
}

std::atomic<uint64_t> g_generation{0};

bool read_file(const std::string& path, std::vector<uint8_t>& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    if (out.size() != kTqLutBytes) {
        err = path + " is " + std::to_string(out.size()) + " bytes, a codebook is " + std::to_string(kTqLutBytes) +
              " (4096 fp16, little-endian)";
        return false;
    }
    return true;
}

}  // namespace

const uint16_t* tq_lut_f16_host() { return state().lut; }
bool tq_lut_builtin() { return state().builtin; }
uint64_t tq_lut_generation() { return g_generation.load(std::memory_order_acquire); }
std::string tq_lut_sha256() { return state().sha; }
std::string tq_lut_builtin_sha256() {
    static const std::string s = sha256_hex(tq_lut_f16, kTqLutBytes);
    return s;
}

bool tq_lut_file_sha256(const std::string& path, std::string& hex, std::string& err) {
    std::vector<uint8_t> b;
    if (!read_file(path, b, err)) return false;
    hex = sha256_hex(b.data(), b.size());
    return true;
}

float tq_lut_max_abs(const uint16_t* f16) {
    float m = 0.0f;
    for (int i = 0; i < kTqLutValues; ++i) m = std::max(m, std::fabs(half_to_float(f16[i])));
    return m;
}

void tq_lut_i8_from_f16(const uint16_t* f16, uint16_t* i8) {
    for (int i = 0; i < kTqLutPoints; ++i) {
        uint8_t q[2];
        for (int j = 0; j < 2; ++j) {
            long r = std::lround(half_to_float(f16[2 * i + j]) / TQ_LUT_I8_SCALE);
            r = r < -127 ? -127 : (r > 127 ? 127 : r);
            q[j] = (uint8_t) (int8_t) r;
        }
        i8[i] = (uint16_t) (q[0] | ((uint16_t) q[1] << 8));
    }
}

bool tq_lut_load(const std::string& path, std::string& err) {
    std::vector<uint8_t> b;
    if (!read_file(path, b, err)) return false;
    uint16_t lut[kTqLutValues];
    std::memcpy(lut, b.data(), kTqLutBytes);   // the file is little-endian, as every supported host
    const float lim = 127.0f * TQ_LUT_I8_SCALE;
    for (int i = 0; i < kTqLutValues; ++i) {
        const float v = half_to_float(lut[i]);
        if (!std::isfinite(v)) { err = path + ": value " + std::to_string(i) + " is not finite"; return false; }
        if (std::fabs(v) > lim) {
            char m[192];
            std::snprintf(m, sizeof m, ": |value %d| = %.6f exceeds %.6f (the int8 copy's range, 127 x TQ_LUT_I8_SCALE)",
                          i, (double) std::fabs(v), (double) lim);
            err = path + m;
            return false;
        }
    }
    // the generator against the built-in pair: the device's int8 copy of this file is made the same way
    std::array<uint16_t, kTqLutPoints> chk{};
    tq_lut_i8_from_f16(tq_lut_f16, chk.data());
    if (std::memcmp(chk.data(), tq_lut_i8, sizeof(tq_lut_i8)) != 0) {
        err = "the int8 codebook generator does not reproduce tq_lut_i8.cuh from the built-in codebook";
        return false;
    }
    State& s = state();
    std::memcpy(s.lut, lut, kTqLutBytes);
    s.builtin = std::memcmp(lut, tq_lut_f16, kTqLutBytes) == 0;
    s.sha = sha256_hex(lut, kTqLutBytes);
    g_generation.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

}  // namespace strata::kernels
