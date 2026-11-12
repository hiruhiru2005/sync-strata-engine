// src/kernels/cpu/sb_pool_bench.cpp - QuantLab (11/13): the decode's CPU experts of an SB pack (bih3 R1Q2, types 180 + K)
// on the engine's own pool, from RAM.  A benchmark, no test.
//
// E experts of Flash-Next's shape (gate/up 640 x 2560, down 2560 x 640, R1Q2 blocks of K bits) are laid out like the RAM
// copy (one arena, blob after blob; 4 KiB pages unless --thp).  Each "layer" takes J distinct random experts with nt tokens
// each and runs them as the engine's dispatch does (ExpertPool::run_split_multi_native: the gate/up phase, the
// intermediate's quantization on the host, the down phase), the pool placed and sized as the engine places it (workers one
// per physical core, the host thread pinned and draining).  The arena is far larger than the L3, so the experts come
// from DRAM.  Reported: the median and mean wall time per layer for each J, and the line through them (us per expert and
// the fixed part per layer) - the numbers STRATA_QL_LAYER_CPU shows in the engine.
//
//   sb_pool_bench [--k 8] [--mb 1024] [--layers 1500] [--j 1,2,3,4] [--nt 1,2] [--workers 0] [--thp]
// The kernels' switches are the engine's environment variables (STRATA_QL_TQ_PRE, STRATA_QL_UNPACK_ONCE, ...).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace cpu = strata::kernels::cpu;

namespace {

std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        v.push_back(std::atoi(s.substr(from, c == std::string::npos ? std::string::npos : c - from).c_str()));
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

uint16_t half_of(float f) {   // positive normal values only (the scales)
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const int e = (int) ((b >> 23) & 255) - 127 + 15;
    return (uint16_t) ((e << 10) | ((b >> 13) & 1023));
}

}  // namespace

int main(int argc, char** argv) {
    int K = 8, layers = 1500, workers = 0;
    size_t mb = 1024;
    bool thp = false;
    std::vector<int> js = {1, 2, 3, 4}, nts = {1, 2};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--k" && more) K = std::atoi(argv[++i]);
        else if (a == "--mb" && more) mb = (size_t) std::atoll(argv[++i]);
        else if (a == "--layers" && more) layers = std::max(10, std::atoi(argv[++i]));
        else if (a == "--j" && more) js = ints(argv[++i]);
        else if (a == "--nt" && more) nts = ints(argv[++i]);
        else if (a == "--workers" && more) workers = std::atoi(argv[++i]);
        else if (a == "--thp") thp = true;
        else {
            std::fprintf(stderr, "usage: sb_pool_bench [--k 8] [--mb 1024] [--layers 1500] [--j 1,2,3,4] [--nt 1,2] [--workers 0] [--thp]\n");
            return 2;
        }
    }
    const int type = cpu::kTqBih3Base + K;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(type, type, cpu::H, cpu::FF, f, err)) { std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); return 1; }
    const float p2[3][4] = {{0.84f, 0.95f, 1.07f, 1.2f}, {0.84f, 0.95f, 1.07f, 1.2f}, {0.84f, 0.95f, 1.07f, 1.2f}};
    if (!cpu::ql_r1q2_fmt(f, p2, err)) { std::fprintf(stderr, "ql_r1q2_fmt: %s\n", err.c_str()); return 1; }
    const size_t E = std::max<size_t>(8, mb * 1000000 / f.bytes);
    const size_t bytes = E * f.bytes, align = thp ? (2u << 20) : 4096u;
    uint8_t* arena = (uint8_t*) std::aligned_alloc(align, (bytes + align - 1) / align * align);
    if (arena == nullptr) { std::fprintf(stderr, "allocation of %zu bytes failed\n", bytes); return 1; }
#if defined(__linux__)
    if (thp && madvise(arena, (bytes + align - 1) / align * align, MADV_HUGEPAGE) != 0) std::fprintf(stderr, "madvise failed\n");
#endif
    std::mt19937_64 rng(1113);
    {   // codes random; per matrix rowbase ~1, colscale ~2^-7, the 2-bit codes random
        uint64_t* p = (uint64_t*) arena;
        for (size_t i = 0; i < bytes / 8; ++i) p[i] = rng();
        for (size_t e = 0; e < E; ++e) {
            uint8_t* b = arena + e * f.bytes + f.sc_off;
            for (int m = 0; m < 3; ++m) {
                uint8_t* s = b + cpu::ql_r1q2_matrix_off(f, m);
                const int64_t rows = m < 2 ? f.n_ff : f.n_embd, nb = (m < 2 ? f.n_embd : f.n_ff) / 128;
                uint16_t* rbase = (uint16_t*) s;
                uint16_t* cs = (uint16_t*) (s + rows * 2);
                for (int64_t r = 0; r < rows; ++r) rbase[r] = half_of(0.9f + 0.2f * (float) (rng() % 1000) / 1000.f);
                for (int64_t c = 0; c < nb; ++c) cs[c] = half_of(0.0078125f * (0.9f + 0.2f * (float) (rng() % 1000) / 1000.f));
            }
        }
    }
    // the window's token activations (as the dispatch's actq makes them) and the output rows
    std::vector<uint8_t> nact((size_t) cpu::MAXT * cpu::kNativeActBytes);
    {
        std::vector<float> x(cpu::H);
        std::normal_distribution<float> g(0.f, 1.f);
        for (int t = 0; t < cpu::MAXT; ++t) {
            for (float& v : x) v = g(rng);
            cpu::native_quant_act(f, x.data(), nact.data() + (size_t) t * cpu::kNativeActBytes);
        }
    }
    std::vector<cpu::ActQ> actq(cpu::MAXT);
    const int jmax = *std::max_element(js.begin(), js.end()), ntmax = *std::max_element(nts.begin(), nts.end());
    std::vector<float> outbuf((size_t) jmax * ntmax * cpu::H);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) jmax);

    // the pool first (it reads the CPUs this thread may use), then the host pinned - the engine's order
    const int host = cpu::planned_host_core(cpu::PoolAffinity::All);
    cpu::ExpertPool pool(workers, /*pin=*/true, /*host_works=*/true, cpu::PoolAffinity::All, 0);
    const cpu::ThreadAffinity prev = cpu::pin_current_thread(host);
    std::printf("sb_pool_bench: K %d (type %d), %zu experts of %.3f MB (%.0f MB, %s pages), %d workers + the host on CPU %d, "
                "trellis rows on %s; STRATA_QL_TQ_PRE=%s STRATA_QL_UNPACK_ONCE=%s STRATA_QL_FUSED_QUANT=%s\n",
                K, type, E, f.bytes / 1e6, bytes / 1e6, thp ? "2 MiB (MADV_HUGEPAGE)" : "4 KiB", pool.workers(), host,
                cpu::tq_isa_name(cpu::tq_isa()), std::getenv("STRATA_QL_TQ_PRE") ? std::getenv("STRATA_QL_TQ_PRE") : "-",
                std::getenv("STRATA_QL_UNPACK_ONCE") ? std::getenv("STRATA_QL_UNPACK_ONCE") : "-",
                std::getenv("STRATA_QL_FUSED_QUANT") ? std::getenv("STRATA_QL_FUSED_QUANT") : "-");
    uint64_t hash = 0xcbf29ce484222325ull;
    for (int nt : nts) {
        std::vector<double> xs, ys;
        for (int J : js) {
            std::vector<double> us;
            us.reserve((size_t) layers);
            for (int l = -50; l < layers; ++l) {   // the first 50 warm the pool up
                std::vector<size_t> pick;
                while ((int) pick.size() < J) {
                    const size_t e = (size_t) (rng() % E);
                    if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                }
                for (int j = 0; j < J; ++j) {
                    cpu::ExpertJobMulti& jb = jobs[(size_t) j];
                    jb = cpu::ExpertJobMulti{};
                    jb.blob = arena + pick[(size_t) j] * f.bytes;
                    jb.nt = nt;
                    for (int t = 0; t < nt; ++t) {
                        const int tok = (j + t) % cpu::MAXT;
                        jb.act[t] = &actq[(size_t) tok];
                        jb.nact[t] = nact.data() + (size_t) tok * cpu::kNativeActBytes;
                        jb.out[t] = outbuf.data() + ((size_t) j * ntmax + t) * cpu::H;
                    }
                }
                const auto t0 = std::chrono::steady_clock::now();
                pool.run_split_multi_native(f, jobs.data(), J, cpu::NativeArithmetic::Default);
                const auto t1 = std::chrono::steady_clock::now();
                if (l >= 0) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                for (int j = 0; j < J; ++j)   // every output bit, so two runs of the same seed can be compared
                    for (int t = 0; t < nt; ++t) {
                        const uint32_t* o = (const uint32_t*) jobs[(size_t) j].out[t];
                        for (int r = 0; r < cpu::H; ++r) hash = (hash ^ o[r]) * 0x100000001b3ull;
                    }
            }
            std::vector<double> s = us;
            std::sort(s.begin(), s.end());
            double mean = 0;
            for (double v : us) mean += v;
            mean /= (double) us.size();
            std::printf("  nt %d J %d: median %7.1f us, mean %7.1f us per layer (p10 %.1f, p90 %.1f)\n", nt, J,
                        s[s.size() / 2], mean, s[s.size() / 10], s[s.size() * 9 / 10]);
            xs.push_back(J);
            ys.push_back(s[s.size() / 2]);
        }
        if (xs.size() >= 2) {   // least squares: median = a + b J
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            const double m = (double) xs.size();
            for (size_t i = 0; i < xs.size(); ++i) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
            const double b = (m * sxy - sx * sy) / (m * sxx - sx * sx), a = (sy - b * sx) / m;
            std::printf("  nt %d: %.1f us per expert + %.1f us per layer (median, least squares of the medians over J)\n", nt, b, a);
        }
    }
    std::printf("output hash %016llx (the same seed and arguments: the same hash when the rows' bits are the same)\n",
                (unsigned long long) hash);
    cpu::restore_thread_affinity(prev);
    std::free(arena);
    return 0;
}
