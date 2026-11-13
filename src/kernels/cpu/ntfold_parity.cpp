// src/kernels/cpu/ntfold_parity.cpp - QuantLab (11/13): STRATA_QL_CPU_NTFOLD's check.  Writes every output float of
// tq_rows_avx512_r1q2_bih3 over K 3..13, nt 1..8, n 2560 / 640, row ranges with odd ends, two random weight sets, to the
// file named on the command line; run it with and without STRATA_QL_CPU_NTFOLD=1 and compare the files (cmp): equal files
// = the same bits.  Needs AVX-512 (exits 77 otherwise).
//   ntfold_parity <out.bin>
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace tqd = strata::kernels::cpu::tq_detail;

namespace {
uint16_t f2h_pos(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const int e = (int) ((b >> 23) & 255) - 127 + 15;
    return (uint16_t) ((e << 10) | ((b >> 13) & 1023));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: ntfold_parity <out.bin>\n"); return 2; }
    if (!cpu::tq_isa_ok(cpu::TqIsa::avx512)) { std::fprintf(stderr, "no AVX-512: skipped\n"); return 77; }
    FILE* fo = std::fopen(argv[1], "wb");
    if (!fo) return 1;
    constexpr int MAXT = cpu::kTqMaxTokens;
    size_t floats = 0;
    for (int seed = 1; seed <= 2; ++seed) {
        std::mt19937_64 rng(1113 + seed);
        for (int n : {2560, 640}) {
            const int rows = n == 2560 ? 640 : 2560, nb = n / 128, qb = (nb + 3) / 4;
            // activations: q8_0 (fp16 d + 32 int8), seed 2 with the int8 extremes often
            std::vector<tqd::TqAct> act(MAXT);
            std::vector<uint8_t> q((size_t) (n / 32) * 34);
            for (int t = 0; t < MAXT; ++t) {
                for (int b = 0; b < n / 32; ++b) {
                    const uint16_t d = f2h_pos(0.001f + 0.05f * (float) (rng() % 1000) / 1000.f);
                    std::memcpy(&q[(size_t) b * 34], &d, 2);
                    for (int j = 0; j < 32; ++j) {
                        int v = (int) (rng() % 255) - 127;
                        if (seed == 2 && rng() % 4 == 0) v = rng() % 2 ? 127 : -127;
                        q[(size_t) b * 34 + 2 + j] = (uint8_t) (int8_t) v;
                    }
                }
                tqd::unpack_act(q.data(), n, act[(size_t) t]);
            }
            const tqd::TqAct* pa[MAXT];
            for (int t = 0; t < MAXT; ++t) pa[t] = &act[(size_t) t];
            std::vector<uint16_t> rbase(rows), cs(nb);
            std::vector<uint8_t> qq((size_t) rows * qb);
            for (auto& v : rbase) v = f2h_pos(0.5f + (float) (rng() % 1000) / 1000.f);
            for (auto& v : cs) v = f2h_pos(0.004f + 0.02f * (float) (rng() % 1000) / 1000.f);
            for (auto& v : qq) v = (uint8_t) rng();
            tqd::TqScales sc;
            sc.rowbase = rbase.data(); sc.colscale = cs.data(); sc.q = qq.data(); sc.qbytes = qb;
            for (int j = 0; j < 4; ++j) sc.p2[j] = std::exp2(-0.21f + 0.145f * j);
            for (int kb = 3; kb <= 13; ++kb) {
                const size_t row_bytes = (size_t) nb * 4 * kb;
                std::vector<uint8_t> w(row_bytes * rows);
                for (auto& v : w) v = (uint8_t) rng();
                std::vector<float> o((size_t) MAXT * rows);
                for (int nt = 1; nt <= MAXT; ++nt)
                    for (int range = 0; range < 3; ++range) {
                        const int r0 = range == 0 ? 0 : range == 1 ? 3 : rows - 1;
                        const int r1 = range == 0 ? rows : range == 1 ? rows - 5 : rows;
                        std::fill(o.begin(), o.end(), -1.0f);
                        float* po[MAXT];
                        for (int t = 0; t < MAXT; ++t) po[t] = o.data() + (size_t) t * rows;
                        tqd::tq_rows_avx512_r1q2_bih3(kb, w.data(), row_bytes, n, pa, nt, po, r0, r1, sc);
                        std::fwrite(o.data(), 4, o.size(), fo);
                        floats += o.size();
                    }
            }
        }
    }
    std::fclose(fo);
    std::printf("ntfold_parity: %zu floats written (STRATA_QL_CPU_NTFOLD=%s)\n", floats,
                std::getenv("STRATA_QL_CPU_NTFOLD") ? std::getenv("STRATA_QL_CPU_NTFOLD") : "-");
    return 0;
}
