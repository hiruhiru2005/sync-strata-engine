// src/kernels/cpu/tqprobe_cpu.cpp - QuantLab (11/13): where the decode's CPU expert time goes (bih3 R1Q2 rows on AVX-512,
// tq_avx512.cpp rows_r1_bih3_t).  A benchmark, no test.  SPEED ONLY for the PROBE rows: their sums are wrong on purpose.
//
//   tqprobe_cpu [--k 8 | --kmix] [--mb 1024] [--layers 1500] [--j 1,2,3,4] [--nt 1,2,4] [--threads 8] [--pv 0,1,2,3,4,5,9]
//
// E experts of Flash-Next's shape (gate/up 640 x 2560, down 2560 x 640, R1Q2 blocks of K bits) in one arena far larger
// than the L3.  A "layer" takes J distinct random experts with nt tokens each; `--threads` threads (one per physical
// core, CPUs 0..threads-1, the calling thread is thread 0) split each matrix's rows evenly: the gate and up rows (and
// the SwiGLU), a barrier, the down rows (on fixed random activations: the intermediate's quantization is not timed), a
// barrier - the shape of the engine's pool (ExpertPool::run_split_multi_native) without its queue.
//   9 ENG     the engine's tq_rows_avx512_r1q2_bih3 (the reference: the copy P0 is checked bit for bit against it)
//   0 REAL    this file's copy of rows_r1_bih3_t
//   1 NOPT    no points: the planes are the state XOR constants (no VPMULLD, no shifts / VPTERNLOG)
//   2 ONEPL   one plane: the three planes XORed into one, one VPDPBUSD per half instead of three (the points stay)
//   3 NOPT1   1 + 2
//   4 STATES  the block's load and the 32 states only (no points, no VPDPBUSD; the per-token tail stays)
//   5 READ    the block's masked load only
// Reported per variant and nt: the median wall per layer for each J, the line through them (us per expert + fixed), and
// the threads' busy share in the phases.
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::kernels::cpu::tq_detail::TqAct;
using strata::kernels::cpu::tq_detail::TqScales;
namespace tqd = strata::kernels::cpu::tq_detail;
constexpr int H = 2560, FF = 640, MAXT = 4;

float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t b;
    if (e == 0) {
        if (m == 0) b = s;
        else { float f = std::ldexp((float) m, -24); std::memcpy(&b, &f, 4); b |= s; }
    } else if (e == 31) b = s | 0x7F800000u | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
uint16_t f2h_pos(float f) {   // positive normal values only
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const int e = (int) ((b >> 23) & 255) - 127 + 15;
    return (uint16_t) ((e << 10) | ((b >> 13) & 1023));
}

#define TQP_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,avx512dq,fma")))
TQP_TARGET inline __m128i reduce_halves(__m512i A, __m512i B) {
    const __m256i x = _mm256_hadd_epi32(_mm512_castsi512_si256(A), _mm512_extracti64x4_epi64(A, 1));
    const __m256i y = _mm256_hadd_epi32(_mm512_castsi512_si256(B), _mm512_extracti64x4_epi64(B, 1));
    const __m256i z = _mm256_hadd_epi32(x, y);
    return _mm_add_epi32(_mm256_castsi256_si128(z), _mm256_extracti128_si256(z, 1));
}
template <int KB>
struct Bih3States {
    alignas(64) uint8_t idx[2][64];
    alignas(64) uint32_t sh[2][16];
    Bih3States() {
        constexpr int NB = 4 * KB;
        for (int g = 0; g < 2; ++g)
            for (int l = 0; l < 16; ++l) {
                const int t = 16 * g + l;
                if (KB == 8) {
                    idx[g][4 * l + 0] = (uint8_t) t;
                    idx[g][4 * l + 1] = (uint8_t) ((t + 31) & 31);
                    idx[g][4 * l + 2] = idx[g][4 * l + 3] = 0;
                    sh[g][l] = 0;
                } else {
                    const int off = (31 - t) * KB, b = off >> 3;
                    for (int j = 0; j < 4; ++j) idx[g][4 * l + j] = (uint8_t) ((b + j) % NB);
                    sh[g][l] = (uint32_t) (off & 7);
                }
            }
    }
};
// tq_avx512.cpp rows_r1_bih3_t with the cut-outs
template <int KB, int P>
TQP_TARGET void rows_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* const* a, int nt, float* const* out, int r0,
                       int r1, const TqScales& sc) {
    const int nb = n / 128;
    constexpr size_t bsz = 4 * KB;
    static const Bih3States<KB> S;
    const __m512i ix0 = _mm512_load_si512(S.idx[0]), ix1 = _mm512_load_si512(S.idx[1]);
    const __m512i sh0 = _mm512_load_si512(S.sh[0]), sh1 = _mm512_load_si512(S.sh[1]);
    const __m512i m15 = _mm512_set1_epi32(0x7FFF);
    const __mmask64 lm = (__mmask64) (bsz >= 64 ? ~0ull : (1ull << bsz) - 1);
    const __m512i c0 = _mm512_set1_epi32((int) 0x9E3779B1u), c1 = _mm512_set1_epi32((int) 0x85EBCA6Bu);
    const __m512i c2 = _mm512_set1_epi32((int) 0xC2B2AE35u), c3 = _mm512_set1_epi32((int) 0x27D4EB2Fu);
    const __m512i m24 = _mm512_set1_epi32((int) 0xFF000000u), m16 = _mm512_set1_epi32((int) 0xFFFF0000u);
    const __m512i m8 = _mm512_set1_epi32((int) 0xFFFFFF00u);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        const uint8_t* qr = sc.q + (size_t) r * sc.qbytes;
        __m128 acc[MAXT];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        __m512i junk = _mm512_setzero_si512();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* qs = x + (size_t) i * bsz;
            const __m512i blk = _mm512_maskz_loadu_epi8(lm, qs);
            if constexpr (P == 5) {
                junk = _mm512_xor_si512(junk, blk);
                continue;
            }
            __m512i u[6];
            for (int g = 0; g < 2; ++g) {
                const __m512i s = _mm512_and_si512(
                    _mm512_srlv_epi32(_mm512_permutexvar_epi8(g ? ix1 : ix0, blk), g ? sh1 : sh0), m15);
                if constexpr (P == 4) {
                    junk = _mm512_xor_si512(junk, s);
                    u[3 * g] = u[3 * g + 1] = u[3 * g + 2] = s;
                } else if constexpr (P == 1 || P == 3) {
                    u[3 * g] = _mm512_xor_si512(s, c0);
                    u[3 * g + 1] = _mm512_xor_si512(s, c1);
                    u[3 * g + 2] = _mm512_xor_si512(s, c2);
                } else {
                    const __m512i x0 = _mm512_mullo_epi32(s, c0), x1 = _mm512_mullo_epi32(s, c1);
                    const __m512i x2 = _mm512_mullo_epi32(s, c2), x3 = _mm512_mullo_epi32(s, c3);
                    u[3 * g] = _mm512_ternarylogic_epi32(m24, _mm512_slli_epi32(x1, 16), _mm512_srli_epi32(x0, 8), 0xEA);
                    u[3 * g + 1] = _mm512_ternarylogic_epi32(m16, _mm512_slli_epi32(x2, 8), _mm512_srli_epi32(x1, 16), 0xEA);
                    u[3 * g + 2] = _mm512_ternarylogic_epi32(m8, x3, _mm512_srli_epi32(x2, 24), 0xEA);
                }
                if constexpr (P == 2 || P == 3) u[3 * g] = _mm512_ternarylogic_epi32(u[3 * g], u[3 * g + 1], u[3 * g + 2], 0x96);
            }
            const __m128 dx = _mm_set1_ps(h2f(sc.colscale[i]) * sc.p2[(qr[i >> 2] >> (2 * (i & 3))) & 3]);
            for (int t = 0; t < nt; ++t) {
                const int8_t* a8 = a[t]->q8 + 128 * i;
                const __m512i qa = _mm512_load_si512(a8), qb = _mm512_load_si512(a8 + 64);
                const __m512i z = _mm512_setzero_si512();
                __m512i A, B;
                if constexpr (P == 4) {
                    A = _mm512_xor_si512(u[0], qa);
                    B = _mm512_xor_si512(u[3], qb);
                } else if constexpr (P == 2 || P == 3) {
                    A = _mm512_dpbusd_epi32(z, u[0], qa);
                    B = _mm512_dpbusd_epi32(z, u[3], qb);
                } else if constexpr (P == 7) {   // the same sums, the chains split (A0 + A1 + A2 apart, added)
                    const __m512i a0 = _mm512_dpbusd_epi32(z, u[0], qa), a1 = _mm512_dpbusd_epi32(z, u[1], qa);
                    const __m512i b0 = _mm512_dpbusd_epi32(z, u[3], qb), b1 = _mm512_dpbusd_epi32(z, u[4], qb);
                    A = _mm512_add_epi32(_mm512_dpbusd_epi32(a0, u[2], qa), a1);
                    B = _mm512_add_epi32(_mm512_dpbusd_epi32(b0, u[5], qb), b1);
                } else {
                    A = _mm512_dpbusd_epi32(z, u[0], qa);
                    A = _mm512_dpbusd_epi32(A, u[1], qa);
                    A = _mm512_dpbusd_epi32(A, u[2], qa);
                    B = _mm512_dpbusd_epi32(z, u[3], qb);
                    B = _mm512_dpbusd_epi32(B, u[4], qb);
                    B = _mm512_dpbusd_epi32(B, u[5], qb);
                }
                if constexpr (P == 6) {   // no tail: the sums folded into one register
                    junk = _mm512_add_epi32(junk, _mm512_add_epi32(A, B));
                    continue;
                }
                const __m128i s1 = _mm_sub_epi32(reduce_halves(A, B), _mm_load_si128((const __m128i*) (a[t]->bsum + 4 * i)));
                const __m128i s = _mm_slli_epi32(s1, 6);
                const __m128 scl = _mm_mul_ps(dx, _mm_loadu_ps(a[t]->d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), scl, acc[t]);
            }
        }
        const float rb = h2f(sc.rowbase[r]);
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = (((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f)) * rb;
            if constexpr (P == 4 || P == 5 || P == 6) out[t][r] += (float) _mm512_reduce_add_epi32(junk);
        }
    }
}
using RowsFn = void (*)(const uint8_t*, size_t, int, const TqAct* const*, int, float* const*, int, int, const TqScales&);
template <int P>
RowsFn rows_for(int kb) {
    switch (kb) {
#define TQP_K(K) case K: return rows_t<K, P>;
        TQP_K(3) TQP_K(4) TQP_K(5) TQP_K(6) TQP_K(7) TQP_K(8) TQP_K(9) TQP_K(10) TQP_K(11) TQP_K(12) TQP_K(13)
#undef TQP_K
        default: return nullptr;
    }
}
RowsFn rows_of(int p, int kb) {
    switch (p) {
        case 0: return rows_for<0>(kb);
        case 1: return rows_for<1>(kb);
        case 2: return rows_for<2>(kb);
        case 3: return rows_for<3>(kb);
        case 4: return rows_for<4>(kb);
        case 5: return rows_for<5>(kb);
        case 6: return rows_for<6>(kb);
        case 7: return rows_for<7>(kb);
        default: return nullptr;
    }
}
const char* pname(int p) {
    static const char* n[] = {"REAL", "NOPT", "ONEPL", "NOPT1", "STATES", "READ", "NOTAIL", "SPLIT", "", "ENG"};
    return n[p];
}

struct Expert {   // one blob: gate codes, up codes, down codes, then each matrix's scales
    int kg, kd;
    const uint8_t* g;
    const uint8_t* u;
    const uint8_t* d;
    size_t gu_row, d_row;
    TqScales sg, su, sd;
};

void pin(int c) {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

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

}  // namespace

int main(int argc, char** argv) {
    int K = 8, layers = 1500, NT = 8;
    std::string mode = "plain";   // 11/13 (4): warm = the layer's experts computed twice, the second timed (L3-hot);
                                  // pf = each thread prefetches its own later rows --pfd KiB ahead; sib = a read-only
                                  // helper on each worker's SMT sibling (CPU + 8) runs through the worker's rows ahead
    size_t pfd = 256 * 1024;
    bool kmix = false;
    size_t mb = 1024;
    std::vector<int> js = {1, 2, 3, 4}, nts = {1, 2, 4}, pv = {9, 0, 1, 2, 3, 4, 5};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--k" && more) K = std::atoi(argv[++i]);
        else if (a == "--kmix") kmix = true;
        else if (a == "--mb" && more) mb = (size_t) std::atoll(argv[++i]);
        else if (a == "--layers" && more) layers = std::max(20, std::atoi(argv[++i]));
        else if (a == "--j" && more) js = ints(argv[++i]);
        else if (a == "--nt" && more) nts = ints(argv[++i]);
        else if (a == "--threads" && more) NT = std::max(1, std::atoi(argv[++i]));
        else if (a == "--pv" && more) pv = ints(argv[++i]);
        else if (a == "--mode" && more) mode = argv[++i];   // plain | warm | pf | sib
        else if (a == "--pfd" && more) pfd = (size_t) std::atoll(argv[++i]) * 1024;
        else {
            std::fprintf(stderr, "usage: tqprobe_cpu [--k 8 | --kmix] [--mb 1024] [--layers 1500] [--j 1,2,3,4] [--nt 1,2,4] [--threads 8] [--pv 9,0,1,2,3,4,5]\n");
            return 2;
        }
    }
    const int kk[10] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const int kw[10] = {158, 756, 1981, 3017, 3563, 4899, 3709, 3118, 1752, 1623};
    std::discrete_distribution<int> kdist(kw, kw + 10);
    std::mt19937_64 rng(1113);
    const int nbg = H / 128, nbd = FF / 128, qge = (nbg + 3) / 4, qdf = (nbd + 3) / 4;
    auto blob_bytes = [&](int kg, int kd) {
        const size_t codes = 2 * (size_t) FF * nbg * 4 * kg + (size_t) H * nbd * 4 * kd;
        const size_t sc = 2 * ((size_t) FF * 2 + nbg * 2 + (size_t) FF * qge) + ((size_t) H * 2 + nbd * 2 + (size_t) H * qdf);
        return (codes + sc + 64 + 4095) / 4096 * 4096;
    };
    std::vector<int> kgs;
    size_t total = 0;
    while (total < mb * 1000000) {
        const int kg = kmix ? kk[kdist(rng)] : K;
        kgs.push_back(kg);
        total += blob_bytes(kg, kg + (kmix ? 1 : 1));
    }
    const size_t E = kgs.size();
    uint8_t* arena = (uint8_t*) std::aligned_alloc(4096, total);
    for (size_t i = 0; i < total / 8; ++i) ((uint64_t*) arena)[i] = rng();
    std::vector<Expert> ex(E);
    {
        size_t off = 0;
        for (size_t e = 0; e < E; ++e) {
            Expert& x = ex[e];
            x.kg = kgs[e];
            x.kd = kmix ? x.kg + 1 : (K + 1 <= 13 ? K + 1 : K);
            x.gu_row = (size_t) nbg * 4 * x.kg;
            x.d_row = (size_t) nbd * 4 * x.kd;
            uint8_t* b = arena + off;
            x.g = b;
            x.u = b + x.gu_row * FF;
            x.d = b + 2 * x.gu_row * FF;
            uint8_t* s = b + 2 * x.gu_row * FF + x.d_row * H;
            auto mk = [&](TqScales& sc, int rows, int nb, int qb) {
                uint16_t* rbase = (uint16_t*) s;
                uint16_t* cs = rbase + rows;
                uint8_t* q = (uint8_t*) (cs + nb);
                for (int r = 0; r < rows; ++r) rbase[r] = f2h_pos(0.9f + 0.2f * (float) (rng() % 1000) / 1000.f);
                for (int c = 0; c < nb; ++c) cs[c] = f2h_pos(0.0078125f * (0.9f + 0.2f * (float) (rng() % 1000) / 1000.f));
                sc.rowbase = rbase; sc.colscale = cs; sc.q = q; sc.qbytes = qb;
                for (int j = 0; j < 4; ++j) sc.p2[j] = 0.84f + 0.12f * j;
                s = q + (size_t) rows * qb;
            };
            mk(x.sg, FF, nbg, qge);
            mk(x.su, FF, nbg, qge);
            mk(x.sd, H, nbd, qdf);
            off += blob_bytes(x.kg, x.kd);
        }
    }
    // activations: q8_0 blocks (fp16 d + 32 int8) of x (n = H) and of h (n = FF), unpacked once (the engine's unpack_all)
    std::vector<TqAct> ax(MAXT), ah(MAXT);
    {
        std::vector<uint8_t> q((size_t) (H / 32) * 34);
        for (int t = 0; t < MAXT; ++t)
            for (int m = 0; m < 2; ++m) {
                const int n = m ? FF : H;
                for (int b = 0; b < n / 32; ++b) {
                    const uint16_t d = f2h_pos(0.01f + 0.02f * (float) (rng() % 1000) / 1000.f);
                    std::memcpy(&q[(size_t) b * 34], &d, 2);
                    for (int j = 0; j < 32; ++j) q[(size_t) b * 34 + 2 + j] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
                }
                tqd::unpack_act(q.data(), n, m ? ah[t] : ax[t]);
            }
    }
    const TqAct* pax[MAXT];
    const TqAct* pah[MAXT];
    for (int t = 0; t < MAXT; ++t) { pax[t] = &ax[t]; pah[t] = &ah[t]; }
    const int JMAX = *std::max_element(js.begin(), js.end());
    std::vector<float> gbuf((size_t) JMAX * MAXT * FF), ubuf(gbuf.size()), hbuf(gbuf.size()), obuf((size_t) JMAX * MAXT * H);

    // the bitwise check: the copy P0 against the engine's rows on a few experts, every row, nt 4
    {
        int bad = 0;
        std::vector<float> r1((size_t) MAXT * H), r2((size_t) MAXT * H);
        for (size_t e = 0; e < std::min<size_t>(E, 24); ++e) {
            const Expert& x = ex[e];
            for (int m = 0; m < 2; ++m) {
                const int rows = m ? H : FF, n = m ? FF : H, kb = m ? x.kd : x.kg;
                float* o1[MAXT];
                float* o2[MAXT];
                for (int t = 0; t < MAXT; ++t) { o1[t] = r1.data() + (size_t) t * H; o2[t] = r2.data() + (size_t) t * H; }
                tqd::tq_rows_avx512_r1q2_bih3(kb, m ? x.d : x.g, m ? x.d_row : x.gu_row, n, m ? pah : pax, MAXT, o1, 0, rows,
                                              m ? x.sd : x.sg);
                rows_of(0, kb)(m ? x.d : x.g, m ? x.d_row : x.gu_row, n, m ? pah : pax, MAXT, o2, 0, rows, m ? x.sd : x.sg);
                for (int t = 0; t < MAXT; ++t) bad += std::memcmp(o1[t], o2[t], 4 * (size_t) rows) != 0;
            }
        }
        std::printf("mode %s (prefetch %zu KiB ahead)\n", mode.c_str(), pfd / 1024);
    std::printf("tqprobe_cpu: %zu experts (%.0f MB, %s), %d threads on CPUs 0..%d; copy REAL vs the engine's rows: %s\n", E,
                    total / 1e6, kmix ? "the SB pack's K mix, down K + 1" : "one K", NT, NT - 1,
                    bad ? "BITS DIFFER" : "bits = engine");
        if (bad) return 1;
    }

    // the pool: NT - 1 workers + this thread, a sense-reversing spin barrier
    std::atomic<int> arrive{0};
    std::atomic<int> sense{0};
    std::atomic<bool> quit{false};
    std::atomic<long> go{0};
    struct Job { int p, J, nt; const int* pick; } job{};
    std::vector<double> busy((size_t) NT, 0.0);
    auto barrier = [&](int& local) {
        local ^= 1;
        if (arrive.fetch_add(1) == NT - 1) { arrive.store(0); sense.store(local, std::memory_order_release); }
        else while (sense.load(std::memory_order_acquire) != local) _mm_pause();
    };
    struct Rng { const uint8_t* p; size_t len; };
    // thread tid's reads in its order: per expert the gate rows [r0, r1), the up rows; then per expert the down rows
    auto stream_of = [&](int tid, std::vector<Rng>& st) {
        st.clear();
        const int g0 = FF * tid / NT, g1 = FF * (tid + 1) / NT, d0 = H * tid / NT, d1 = H * (tid + 1) / NT;
        for (int j = 0; j < job.J; ++j) {
            const Expert& x = ex[(size_t) job.pick[j]];
            st.push_back({x.g + (size_t) g0 * x.gu_row, (size_t) (g1 - g0) * x.gu_row});
            st.push_back({x.u + (size_t) g0 * x.gu_row, (size_t) (g1 - g0) * x.gu_row});
        }
        for (int j = 0; j < job.J; ++j) {
            const Expert& x = ex[(size_t) job.pick[j]];
            st.push_back({x.d + (size_t) d0 * x.d_row, (size_t) (d1 - d0) * x.d_row});
        }
    };
    const bool pf = mode == "pf";
    auto work = [&](int tid, int& local) {
        const int J = job.J, nt = job.nt, p = job.p;
        thread_local std::vector<Rng> st;
        size_t pfi = 0, pfo = 0, pbytes = 0, cbytes = 0;
        if (pf) stream_of(tid, st);
        auto advance = [&](size_t consumed) {   // keep the prefetches pfd bytes ahead of the reads
            cbytes += consumed;
            while (pfi < st.size() && pbytes < cbytes + pfd) {
                _mm_prefetch((const char*) st[pfi].p + pfo, _MM_HINT_T1);
                pfo += 64; pbytes += 64;
                if (pfo >= st[pfi].len) { pfo = 0; ++pfi; }
            }
        };
        if (pf) advance(0);
        auto rows = [&](int kb, const uint8_t* w, size_t rb, int n, const TqAct* const* a, float* const* o, int r0, int r1,
                        const TqScales& sc, int chunk) {
            if (!pf) {
                if (p == 9) tqd::tq_rows_avx512_r1q2_bih3(kb, w, rb, n, a, nt, o, r0, r1, sc);
                else rows_of(p, kb)(w, rb, n, a, nt, o, r0, r1, sc);
                return;
            }
            for (int r = r0; r < r1; r += chunk) {
                const int re = std::min(r1, r + chunk);
                if (p == 9) tqd::tq_rows_avx512_r1q2_bih3(kb, w, rb, n, a, nt, o, r, re, sc);
                else rows_of(p, kb)(w, rb, n, a, nt, o, r, re, sc);
                advance((size_t) (re - r) * rb);
            }
        };
        const auto t0 = std::chrono::steady_clock::now();
        for (int j = 0; j < J; ++j) {   // gate / up rows [r0, r1), the SwiGLU
            const Expert& x = ex[(size_t) job.pick[j]];
            const int r0 = FF * tid / NT, r1 = FF * (tid + 1) / NT;
            float* gp[MAXT];
            float* up[MAXT];
            for (int t = 0; t < nt; ++t) { gp[t] = gbuf.data() + ((size_t) j * MAXT + t) * FF; up[t] = ubuf.data() + ((size_t) j * MAXT + t) * FF; }
            rows(x.kg, x.g, x.gu_row, H, pax, gp, r0, r1, x.sg, 4);
            rows(x.kg, x.u, x.gu_row, H, pax, up, r0, r1, x.su, 4);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    const float g = gp[t][r], u = up[t][r];
                    hbuf[((size_t) j * MAXT + t) * FF + r] = (g / (1.f + std::exp(-g))) * u;
                }
        }
        const auto t1 = std::chrono::steady_clock::now();
        barrier(local);
        const auto t2 = std::chrono::steady_clock::now();
        for (int j = 0; j < J; ++j) {   // down rows
            const Expert& x = ex[(size_t) job.pick[j]];
            const int r0 = H * tid / NT, r1 = H * (tid + 1) / NT;
            float* op[MAXT];
            for (int t = 0; t < nt; ++t) op[t] = obuf.data() + ((size_t) j * MAXT + t) * H;
            rows(x.kd, x.d, x.d_row, FF, pah, op, r0, r1, x.sd, 16);
        }
        const auto t3 = std::chrono::steady_clock::now();
        busy[(size_t) tid] += std::chrono::duration<double, std::micro>((t1 - t0) + (t3 - t2)).count();
        barrier(local);
    };
    // sib: the helpers (CPU NT + h for worker h): each runs through worker h's reads once per layer, a load a line
    std::atomic<uint64_t> hsink{0};
    std::vector<std::thread> helpers;
    if (mode == "sib")
        for (int h = 0; h < NT; ++h)
            helpers.emplace_back([&, h] {
                pin(NT + h);
                long seen = 0;
                std::vector<Rng> hs;
                uint64_t acc = 0;
                for (;;) {
                    long g;
                    while ((g = go.load(std::memory_order_acquire)) == seen && !quit.load(std::memory_order_relaxed)) _mm_pause();
                    if (quit.load()) break;
                    seen = g;
                    stream_of(h, hs);
                    for (const Rng& r : hs) {
                        for (size_t o = 0; o < r.len; o += 64) acc += *(const volatile uint8_t*) (r.p + o);
                        if (go.load(std::memory_order_relaxed) != seen) break;
                    }
                }
                hsink += acc;
            });
    std::vector<std::thread> th;
    for (int w = 1; w < NT; ++w)
        th.emplace_back([&, w] {
            pin(w);
            int local = 0;
            long seen = 0;
            for (;;) {
                long g;
                while ((g = go.load(std::memory_order_acquire)) == seen && !quit.load(std::memory_order_relaxed)) _mm_pause();
                if (quit.load()) return;
                seen = g;
                work(w, local);
            }
        });
    pin(0);
    int local0 = 0;
    long gen = 0;
    std::vector<int> pick((size_t) JMAX);
    auto layer = [&](int p, int J, int nt) {
        for (int j = 0; j < J; ++j) {
            int e;
            do e = (int) (rng() % E); while (std::find(pick.begin(), pick.begin() + j, e) != pick.begin() + j);
            pick[(size_t) j] = e;
        }
        job = Job{p, J, nt, pick.data()};
        if (mode == "warm") {   // the same experts once untimed: the timed pass reads them from the L3
            go.store(++gen, std::memory_order_release);
            work(0, local0);
        }
        const auto t0 = std::chrono::steady_clock::now();
        go.store(++gen, std::memory_order_release);
        work(0, local0);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    };
    for (int nt : nts) {
        for (int p : pv) {
            std::vector<double> xs, ys;
            std::printf("  nt %d P%d %-6s", nt, p, pname(p));
            double bsum = 0, wsum = 0;
            for (int J : js) {
                std::vector<double> us;
                for (int l = -50; l < layers; ++l) {
                    if (l == 0) std::fill(busy.begin(), busy.end(), 0.0);
                    const double u = layer(p, J, nt);
                    if (l >= 0) us.push_back(u);
                }
                std::sort(us.begin(), us.end());
                double w = 0;
                for (double v : us) w += v;
                double b = 0;
                for (double v : busy) b += v;
                bsum += b; wsum += w;
                std::printf("  J%d %6.1f", J, us[us.size() / 2]);
                xs.push_back(J);
                ys.push_back(us[us.size() / 2]);
            }
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            const double m = (double) xs.size();
            for (size_t i = 0; i < xs.size(); ++i) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
            const double b = m > 1 ? (m * sxy - sx * sy) / (m * sxx - sx * sx) : 0, a = (sy - b * sx) / m;
            std::printf("   -> %6.1f us per expert + %5.1f us per layer; busy %.2f of %d threads\n", b, a, bsum / wsum, NT);
            std::fflush(stdout);
        }
    }
    quit.store(true);
    for (auto& t : th) t.join();
    for (auto& t : helpers) t.join();
    std::free(arena);
    return 0;
}
