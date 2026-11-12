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
// The kernels' switches are the engine's environment variables (STRATA_QL_UNPACK_ONCE, ...).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <immintrin.h>
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

// "9-15" or "8,9,12-13"
std::vector<int> cpus_of(const std::string& s) {
    std::vector<int> v;
    size_t from = 0;
    for (;;) {
        const size_t c = s.find(',', from);
        const std::string part = s.substr(from, c == std::string::npos ? std::string::npos : c - from);
        const size_t d = part.find('-');
        if (d == std::string::npos) v.push_back(std::atoi(part.c_str()));
        else for (int x = std::atoi(part.substr(0, d).c_str()); x <= std::atoi(part.substr(d + 1).c_str()); ++x) v.push_back(x);
        if (c == std::string::npos) break;
        from = c + 1;
    }
    return v;
}

std::atomic<uint64_t> sink{0};
inline void strata_bench_pause() { _mm_pause(); }
// a co-runner's AVX-512 VNNI work (the kind the workers do)
__attribute__((target("avx512f,avx512bw,avx512vnni"))) void vec_work() {
    __m512i a = _mm512_set1_epi32(1), b = _mm512_set1_epi32(3), c0 = _mm512_setzero_si512(), c1 = c0, c2 = c0, c3 = c0;
    for (int i = 0; i < 4096; ++i) {
        c0 = _mm512_dpbusd_epi32(c0, a, b); c1 = _mm512_dpbusd_epi32(c1, b, a);
        c2 = _mm512_dpbusd_epi32(c2, a, a); c3 = _mm512_dpbusd_epi32(c3, b, b);
        a = _mm512_add_epi32(a, c0);
    }
    sink.fetch_add((uint64_t) _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0, c1), _mm512_add_epi32(c2, c3))),
                   std::memory_order_relaxed);
}

}  // namespace

int main(int argc, char** argv) {
    int K = 8, layers = 1500, workers = 0;
    size_t mb = 1024;
    bool thp = false, mix = false, host_work = true;
    int gap_us = 0;
    std::string sib = "spin";
    std::vector<int> sib_cpus, bw_cpus;
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
        else if (a == "--mix") mix = true;
        else if (a == "--no-host-work") host_work = false;   // the engine's --no-host-worker
        else if (a == "--gap-us" && more) gap_us = std::atoi(argv[++i]);
        else if (a == "--sib" && more) sib = argv[++i];
        else if (a == "--sib-cpus" && more) sib_cpus = cpus_of(argv[++i]);
        else if (a == "--bw-cpus" && more) bw_cpus = cpus_of(argv[++i]);
        else {
            std::fprintf(stderr, "usage: sb_pool_bench [--k 8] [--mb 1024] [--layers 1500] [--j 1,2,3,4] [--nt 1,2] [--workers 0] [--thp] [--mix] [--gap-us G] [--no-host-work]\n"
                                 "       [--sib spin|alu|vec|copy --sib-cpus 9-15] [--bw-cpus 8]\n");
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
    const int jmax = mix ? 8 : *std::max_element(js.begin(), js.end());
    const int ntmax = mix ? 4 : *std::max_element(nts.begin(), nts.end());
    std::vector<float> outbuf((size_t) jmax * ntmax * cpu::H);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) jmax);

    // the pool first (it reads the CPUs this thread may use), then the host pinned - the engine's order
    const int host = cpu::planned_host_core(cpu::PoolAffinity::All);
    cpu::ExpertPool pool(workers, /*pin=*/true, host_work, cpu::PoolAffinity::All, 0);
    const cpu::ThreadAffinity prev = cpu::pin_current_thread(host);
    std::printf("sb_pool_bench: K %d (type %d), %zu experts of %.3f MB (%.0f MB, %s pages), %d workers + the host on CPU %d, "
                "trellis rows on %s; STRATA_QL_UNPACK_ONCE=%s STRATA_QL_FUSED_QUANT=%s\n",
                K, type, E, f.bytes / 1e6, bytes / 1e6, thp ? "2 MiB (MADV_HUGEPAGE)" : "4 KiB", pool.workers(), host,
                cpu::tq_isa_name(cpu::tq_isa()),
                std::getenv("STRATA_QL_UNPACK_ONCE") ? std::getenv("STRATA_QL_UNPACK_ONCE") : "-",
                std::getenv("STRATA_QL_FUSED_QUANT") ? std::getenv("STRATA_QL_FUSED_QUANT") : "-");
    // QuantLab (11/13) the engine's surroundings, each opt-in: --sib MODE on --sib-cpus (co-runners on the workers' SMT
    // siblings: spin = a pause loop, alu = scalar integer work, vec = AVX-512 VNNI work, copy = memcpy), --bw-cpus (memcpy
    // threads: DRAM traffic like the GPU's reads of the RAM copy), --gap-us (the host spins between layers, as it waits
    // for the GPU there)
    std::atomic<bool> stop{false};
    std::vector<std::thread> extra;
    auto copier = [&](int c) {
        (void) cpu::pin_current_thread(c);
        const size_t sz = 64u << 20;
        std::vector<uint8_t> a(sz, 1), b(sz, 2);
        while (!stop.load(std::memory_order_relaxed)) { std::memcpy(b.data(), a.data(), sz); std::memcpy(a.data(), b.data(), sz); }
    };
    for (int c : sib_cpus)
        extra.emplace_back([&, c] {
            if (sib == "copy") { copier(c); return; }
            (void) cpu::pin_current_thread(c);
            if (sib == "vec") { while (!stop.load(std::memory_order_relaxed)) vec_work(); return; }
            uint64_t x = 1, y = 2, z = 3, w = 4;
            while (!stop.load(std::memory_order_relaxed)) {
                if (sib == "spin") { for (int i = 0; i < 64; ++i) strata_bench_pause(); continue; }
                for (int i = 0; i < 4096; ++i) { x = x * 0x9E3779B1u + y; y = y * 0x85EBCA6Bu + z; z = z * 0xC2B2AE35u + w; w = w * 0x27D4EB2Fu + x; }
                sink.fetch_add(x ^ y ^ z ^ w, std::memory_order_relaxed);
            }
        });
    for (int c : bw_cpus) extra.emplace_back([&, c] { copier(c); });
    std::printf("  surroundings: sib %s on %zu CPUs, %zu copy threads, gap %d us between layers%s\n", sib_cpus.empty() ? "-" : sib.c_str(),
                sib_cpus.size(), bw_cpus.size(), gap_us, mix ? ", the engine's mix of experts per layer and tokens per expert" : "");
    uint64_t hash = 0xcbf29ce484222325ull;
    // one layer: J distinct random experts, expert j with nt[j] tokens; returns the pool's wall time
    auto layer = [&](int J, const int* ntj) {
        std::vector<size_t> pick;
        while ((int) pick.size() < J) {
            const size_t e = (size_t) (rng() % E);
            if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
        }
        for (int j = 0; j < J; ++j) {
            cpu::ExpertJobMulti& jb = jobs[(size_t) j];
            jb = cpu::ExpertJobMulti{};
            jb.blob = arena + pick[(size_t) j] * f.bytes;
            jb.nt = ntj[j];
            for (int t = 0; t < ntj[j]; ++t) {
                const int tok = (j + t) % cpu::MAXT;
                jb.act[t] = &actq[(size_t) tok];
                jb.nact[t] = nact.data() + (size_t) tok * cpu::kNativeActBytes;
                jb.out[t] = outbuf.data() + ((size_t) j * ntmax + t) * cpu::H;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        pool.run_split_multi_native(f, jobs.data(), J, cpu::NativeArithmetic::Default);
        const auto t1 = std::chrono::steady_clock::now();
        for (int j = 0; j < J; ++j)   // every output bit, so two runs of the same seed can be compared
            for (int t = 0; t < ntj[j]; ++t) {
                const uint32_t* o = (const uint32_t*) jobs[(size_t) j].out[t];
                for (int r = 0; r < cpu::H; ++r) hash = (hash ^ o[r]) * 0x100000001b3ull;
            }
        if (gap_us > 0) {
            const auto g = std::chrono::steady_clock::now() + std::chrono::microseconds(gap_us);
            while (std::chrono::steady_clock::now() < g) strata_bench_pause();
        }
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };
    // the pool's busy thread time (STRATA_QL_LAYER_CPU=1 counts it) and phase walls since the last call
    long long b0 = 0;
    double w0 = 0, q0 = 0;
    auto busy_since = [&](double& wall_ms, double& quant_ms) {
        const long long b = pool.ql_busy_gu_ns.load() + pool.ql_busy_down_ns.load();
        const double w = pool.ms_multi_gu + pool.ms_multi_down, q = pool.ms_multi_q;
        const long long d = b - b0;
        wall_ms = w - w0; quant_ms = q - q0;
        b0 = b; w0 = w; q0 = q;
        return d;
    };
    const int threads = pool.workers() + (host_work ? 1 : 0);
    if (mix) {
        // QuantLab (11/13): linux_int1_b26_pf0.6_lcpuA1.log - experts per CPU dispatch (mean ~2.7, 1 to ~10) and tokens
        // per expert ("jobs by tokens": 1 64%, 2 25%, 3 8%, 4 3%)
        const double pj[8] = {0.24, 0.24, 0.19, 0.13, 0.09, 0.06, 0.03, 0.02};
        const double pt[4] = {0.64, 0.25, 0.08, 0.03};
        auto draw = [&](const double* p, int n) {
            double u = (double) (rng() % 1000000) / 1e6;
            for (int i = 0; i < n; ++i) { if (u < p[i]) return i + 1; u -= p[i]; }
            return n;
        };
        std::vector<double> xs, ys;
        double sum_us = 0;
        long long experts = 0, entries = 0;
        int ntj[8];
        for (int l = -50; l < layers; ++l) {
            const int J = draw(pj, 8);
            for (int j = 0; j < J; ++j) ntj[j] = draw(pt, 4);
            if (l == 0) { double a, b; (void) busy_since(a, b); }
            const double us = layer(J, ntj);
            if (l < 0) continue;
            sum_us += us; experts += J;
            for (int j = 0; j < J; ++j) entries += ntj[j];
            xs.push_back(J); ys.push_back(us);
        }
        double wall_ms = 0, quant_ms = 0;
        const long long busy = busy_since(wall_ms, quant_ms);
        const double m = (double) xs.size();
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (size_t i = 0; i < xs.size(); ++i) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
        const double b = (m * sxy - sx * sy) / (m * sxx - sx * sx), a = (sy - b * sx) / m;
        std::printf("  mix: %d layers, %.2f experts and %.2f tokens per expert a layer: %.1f us per layer, %.1f us per expert "
                    "(fit %.1f us per expert + %.1f us per layer); busy %.0f thread-us per expert, %.2f of %d threads busy in "
                    "the phases, quant %.1f us per layer\n", layers, experts / m, (double) entries / experts, sum_us / m,
                    sum_us / experts, b, a, busy / 1e3 / experts, wall_ms > 0 ? busy / 1e6 / wall_ms : 0.0, threads,
                    quant_ms * 1e3 / m);
    } else {
        for (int nt : nts) {
            std::vector<double> xs, ys;
            const int ntj[8] = {nt, nt, nt, nt, nt, nt, nt, nt};
            for (int J : js) {
                std::vector<double> us;
                us.reserve((size_t) layers);
                double wall_ms = 0, quant_ms = 0;
                for (int l = -50; l < layers; ++l) {   // the first 50 warm the pool up
                    if (l == 0) (void) busy_since(wall_ms, quant_ms);
                    const double u = layer(J, ntj);
                    if (l >= 0) us.push_back(u);
                }
                const long long busy = busy_since(wall_ms, quant_ms);
                std::vector<double> s = us;
                std::sort(s.begin(), s.end());
                double mean = 0;
                for (double v : us) mean += v;
                mean /= (double) us.size();
                std::printf("  nt %d J %d: median %7.1f us, mean %7.1f us per layer (p10 %.1f, p90 %.1f); busy %.0f thread-us "
                            "per expert, %.2f of %d threads busy\n", nt, J, s[s.size() / 2], mean, s[s.size() / 10],
                            s[s.size() * 9 / 10], busy / 1e3 / ((double) J * layers),
                            wall_ms > 0 ? busy / 1e6 / wall_ms : 0.0, threads);
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
    }
    stop.store(true);
    for (auto& t : extra) t.join();
    std::printf("output hash %016llx (the same seed and arguments: the same hash when the rows' bits are the same)\n",
                (unsigned long long) hash);
    cpu::restore_thread_affinity(prev);
    std::free(arena);
    return 0;
}
