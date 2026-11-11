// src/kernels/cpu/tqp_cpu_test.cpp - the trellis CPU rows (tq_cpu.hpp) on synthetic blocks, with no ggml and no model:
//
//   * the pair-trellis types TQP5 / TQP6 / TQP7 (ggml 147-149, Gyro-L): the scalar kernel against a reference written
//     here from the format definition (state t = the 16 stream bits that start at bit (63 - t) * K, LSB first,
//     circular over 64 * K bits; the step's two weights are the first pair of the state's codebook entry), summed in
//     double; then every ISA this CPU has (AVX2, AVX-512 takes the AVX2 pair kernel) bit for bit equal to the scalar
//     kernel, for groups of 1..8 tokens, one row range or two, and for the fused gate/up rows;
//   * the older trellis types (144-146): AVX2 / AVX-512 bit for bit equal to the scalar kernel (the pair types were
//     added next to them: this shows they still agree), and their block sizes and K unchanged;
//   * the type predicates (tq_supported, tq_kbits, tq_block_bytes) for every id 0..299: the answers the code gave
//     before the pair types, plus 147-149.
//
// Random stream bytes are valid codes (any 15-bit state indexes the codebook).  CPU only.  Exit 0 = pass.
#include "strata/kernels/cpu/tq_cpu.hpp"
#include "strata/kernels/tq_lut.hpp"

#include "tq_states.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels::cpu;

namespace {

int g_fail = 0;

void fail(const char* what, int a = 0, int b = 0) {
    std::printf("  FAIL: %s %d %d\n", what, a, b);
    ++g_fail;
}

// ---- the type predicates as they were before the pair types (the old formulas, copied)
// QuantLab: TQK5 (150), TQK4 (151) and the local pseudo types kTqPseudoBase + K (K 2, 3, 9..14) are trellis types too
int ql_kbits(int t) {
    if (t == 150) return 5;
    if (t == 151) return 4;
    const int k = t - kTqPseudoBase;
    return ((k >= 2 && k <= 3) || (k >= 9 && k <= 14)) ? k : 0;
}
bool old_supported(int t) {
    return t == 144 || t == 145 || t == 146 || t == kTqTypeK5X || t == kTqTypeK4X || t == kTqTypeK5T || t == kTqTypeK4T ||
           ql_kbits(t) != 0 || (t >= 182 && t <= 194);   // QuantLab (11/12): bih3
}
int old_kbits(int t) {
    return t == 144 ? 8 : t == 145 ? 6 : t == 146 ? 7 : (t == kTqTypeK5X || t == kTqTypeK5T) ? 5 : (t == kTqTypeK4X || t == kTqTypeK4T) ? 4 : (t >= 182 && t <= 194) ? t - 180 : ql_kbits(t);
}

void check_predicates() {
    for (int t = 0; t < 300; ++t) {
        const bool pair = t >= 147 && t <= 149;
        if (tq_supported(t) != (old_supported(t) || pair)) fail("tq_supported", t);
        const int want_k = pair ? (t == 147 ? 5 : t == 148 ? 6 : 7) : old_kbits(t);
        if (tq_kbits(t) != want_k) fail("tq_kbits", t, tq_kbits(t));
        const size_t want_b = pair ? (size_t) (2 + 8 * want_k) : (size_t) (2 + 4 * want_k);
        if (tq_block_bytes(t) != want_b) fail("tq_block_bytes", t, (int) tq_block_bytes(t));
    }
    if (tq_block_bytes(144) != 34 || tq_block_bytes(145) != 26 || tq_block_bytes(146) != 30) fail("sizes of 144-146");
    if (tq_block_bytes(147) != 42 || tq_block_bytes(148) != 50 || tq_block_bytes(149) != 58) fail("sizes of 147-149");
    std::printf("predicates (ids 0..299): %s\n", g_fail ? "FAIL" : "ok");
}

// ---- synthetic data
uint16_t random_half(std::mt19937& rng) {
    std::uniform_int_distribution<int> ex(2, 8), man(0, 1023), sgn(0, 3);
    return (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));   // 2^-13 .. 2^-6
}

std::vector<uint8_t> random_rows(int type, int rows, int n, std::mt19937& rng, size_t& row_bytes) {
    const size_t bs = tq_block_bytes(type);
    row_bytes = bs * (size_t) (n / 128);
    std::vector<uint8_t> w(row_bytes * (size_t) rows);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = random_half(rng);
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}

// n values as Q8_0 blocks (fp16 d, 32 x int8): 34 bytes per 32 values
std::vector<uint8_t> random_q8_0(int n, std::mt19937& rng, std::vector<float>& d, std::vector<int>& q) {
    std::vector<uint8_t> a((size_t) (n / 32) * 34);
    std::uniform_int_distribution<int> val(-127, 127), ex(8, 13), man(0, 1023);
    d.assign((size_t) n / 32, 0.f);
    q.assign((size_t) n, 0);
    for (int b = 0; b < n / 32; ++b) {
        const uint16_t h = (uint16_t) ((ex(rng) << 10) | man(rng));
        std::memcpy(&a[(size_t) b * 34], &h, 2);
        d[(size_t) b] = strata::kernels::cpu::tq_detail::tq_half_to_float(h);
        for (int j = 0; j < 32; ++j) {
            const int v = val(rng);
            q[(size_t) b * 32 + j] = v;
            a[(size_t) b * 34 + 2 + j] = (uint8_t) (int8_t) v;
        }
    }
    return a;
}

// ---- the reference for the pair types, from the format definition (bit by bit, not the window code of the kernels)
uint32_t ref_state(const uint8_t* qs, int K, int t) {
    const int nbits = 64 * K;
    uint32_t s = 0;
    for (int b = 0; b < 16; ++b) {
        const int pos = ((63 - t) * K + b) % nbits;
        s |= (uint32_t) ((qs[pos >> 3] >> (pos & 7)) & 1) << b;
    }
    return s & 0x7FFFu;
}

// w . a in double; `bound` = the sum of the absolute terms (the scale of the float rounding allowed)
double ref_dot(int type, const uint8_t* row, int n, const std::vector<float>& d, const std::vector<int>& q, double& bound) {
    const int K = tq_kbits(type);
    const int16_t* tab = tq_state_table();
    const size_t bs = tq_block_bytes(type);
    double sum = 0, abs_sum = 0;
    for (int i = 0; i < n / 128; ++i) {
        const uint8_t* blk = row + (size_t) i * bs;
        uint16_t hd;
        std::memcpy(&hd, blk, 2);
        const double dw = strata::kernels::cpu::tq_detail::tq_half_to_float(hd);
        uint32_t st[64];
        for (int t = 0; t < 64; ++t) st[t] = ref_state(blk + 2, K, t);
        for (int k = 0; k < 4; ++k) {
            long long s = 0, sa = 0;
            for (int j = 0; j < 32; ++j) {
                const int v = 32 * k + j;                    // weight v of the block: step v / 2, element v & 1 of the first pair
                const int w16 = tab[4 * st[v >> 1] + (v & 1)];
                s += (long long) w16 * q[(size_t) 128 * i + v];
                sa += (long long) std::abs(w16) * std::abs(q[(size_t) 128 * i + v]);
            }
            const double sc = dw * d[(size_t) 4 * i + k];
            sum += sc * (double) s;
            abs_sum += std::fabs(sc) * (double) sa;
        }
    }
    bound = abs_sum / 8192.0;
    return sum / 8192.0;
}

bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

constexpr int NT = kTqMaxTokens;

struct Case {
    int type, rows, n;
    std::vector<uint8_t> w;
    size_t rb = 0;
    std::vector<std::vector<uint8_t>> act;
    std::vector<std::vector<float>> d;
    std::vector<std::vector<int>> q;
    const void* ap[NT];
    Case(int type_, int rows_, int n_, uint32_t seed) : type(type_), rows(rows_), n(n_) {
        std::mt19937 rng(seed);
        w = random_rows(type, rows, n, rng, rb);
        act.resize(NT); d.resize(NT); q.resize(NT);
        for (int t = 0; t < NT; ++t) {
            act[t] = random_q8_0(n, rng, d[t], q[t]);
            ap[t] = act[t].data();
        }
    }
    void run(TqIsa isa, int nt, const void* const* a, std::vector<float>& o) const {
        o.assign((size_t) nt * rows, 0.f);
        float* op[NT];
        for (int t = 0; t < nt; ++t) op[t] = o.data() + (size_t) t * rows;
        tq_rows(isa, type, w.data(), rb, n, a, nt, op, 0, rows / 3);       // two row ranges, as the pool splits them
        tq_rows(isa, type, w.data(), rb, n, a, nt, op, rows / 3, rows);
    }
};

void check_type(int type, bool reference) {
    Case c(type, 37, 1280, 1000u + (uint32_t) type);
    std::vector<float> ref, got;
    c.run(TqIsa::scalar, NT, c.ap, ref);
    int bad = 0;
    if (reference) {   // the pair types: the scalar kernel against the definition
        double worst = 0;
        for (int t = 0; t < NT; ++t)
            for (int r = 0; r < c.rows; ++r) {
                double bound = 0;
                const double want = ref_dot(type, c.w.data() + (size_t) r * c.rb, c.n, c.d[t], c.q[t], bound);
                const double err = std::fabs((double) ref[(size_t) t * c.rows + r] - want);
                worst = std::fmax(worst, bound > 0 ? err / bound : err);
                if (err > 2e-5 * bound + 1e-30) ++bad;
            }
        std::printf("type %d scalar vs the definition (double): %d of %d outside 2e-5 of the term sum, worst %.2e\n", type, bad,
                    NT * c.rows, worst);
        if (bad) fail("scalar kernel differs from the definition", type);
    }
    for (TqIsa isa : {TqIsa::avx2, TqIsa::avx2vnni, TqIsa::avx2e, TqIsa::avx512}) {
        if (!tq_isa_ok(isa)) { std::printf("type %d %-6s not on this CPU: skipped\n", type, tq_isa_name(isa)); continue; }
        bool ok = true;
        for (int nt = 1; nt <= NT; ++nt)
            for (int t0 = 0; t0 + nt <= NT; t0 += nt) {
                c.run(isa, nt, c.ap + t0, got);
                ok = ok && same_bits(got.data(), ref.data() + (size_t) t0 * c.rows, (size_t) nt * c.rows);
            }
        std::printf("type %d %-6s groups of 1..%d tokens vs the scalar kernel: %s\n", type, tq_isa_name(isa), NT,
                    ok ? "bitwise equal" : "DIFFERENT");
        if (!ok) fail("ISA differs from scalar", type);
    }
    // the fused gate/up rows: gate rows at the blob, up rows after them
    {
        std::vector<uint8_t> blob = c.w;
        std::mt19937 rng(77u + (uint32_t) type);
        size_t rb2 = 0;
        const auto up = random_rows(type, c.rows, c.n, rng, rb2);
        blob.insert(blob.end(), up.begin(), up.end());
        const size_t up_off = c.w.size();
        std::vector<float> ff0((size_t) NT * c.rows), ff1((size_t) NT * c.rows);
        float* p0[NT]; float* p1[NT];
        for (int t = 0; t < NT; ++t) { p0[t] = ff0.data() + (size_t) t * c.rows; p1[t] = ff1.data() + (size_t) t * c.rows; }
        tq_gu_rows(TqIsa::scalar, type, blob.data(), c.rb, up_off, c.n, c.ap, NT, p0, 0, c.rows);
        bool ok = true;
        for (TqIsa isa : {TqIsa::avx2, TqIsa::avx2vnni, TqIsa::avx2e, TqIsa::avx512}) {
            if (!tq_isa_ok(isa)) continue;
            tq_gu_rows(isa, type, blob.data(), c.rb, up_off, c.n, c.ap, NT, p1, 0, c.rows);
            ok = ok && same_bits(ff0.data(), ff1.data(), ff0.size());
        }
        std::printf("type %d gate/up rows, every ISA vs scalar: %s\n", type, ok ? "bitwise equal" : "DIFFERENT");
        if (!ok) fail("gate/up rows differ", type);
    }
}

// ---- QuantLab (10/31): R1Q2 blocks (4K code bytes; d = colscale[b] * p2[q[r][b]], rowbase[r] at the end) against the
// same codes in the old layout with d = fp16(colscale * p2) per block and the row's result times rowbase: equal up to the
// fp16 rounding of d (relative 2^-11 a block).  Every ISA bitwise equal to the scalar R1Q2 kernel, fused gate/up too.
uint16_t to_half(float f) {
    // round to nearest even through the float bits (normal positive values only here)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e <= 0 || e >= 31) return (uint16_t) sign;   // not used for out-of-range values
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

void check_r1q2(int type, int n) {
    const int K = tq_kbits(type), nb = n / 128, rows = 37, qbytes = (nb + 3) / 4;
    std::mt19937 rng(4242u + (uint32_t) type + (uint32_t) n);
    std::uniform_int_distribution<int> byte(0, 255), qv(0, 3), ex(13, 16), man(0, 1023);
    const size_t cb = (size_t) 4 * K, rb_new = cb * nb, rb_old = (2 + cb) * nb;
    std::vector<uint8_t> wn(rb_new * rows), wo(rb_old * rows), q((size_t) qbytes * rows, 0);
    std::vector<uint16_t> rbase(rows), cscale(nb);
    tq_detail::TqScales sc;
    sc.p2[0] = 0.5f; sc.p2[1] = 0.7071067f; sc.p2[2] = 1.0f; sc.p2[3] = 1.4142135f;
    for (auto& h : rbase) h = (uint16_t) ((ex(rng) << 10) | man(rng));   // 0.5 .. 4
    for (auto& h : cscale) h = (uint16_t) (((ex(rng) - 4) << 10) | man(rng));   // ~ 2^-6 .. 2^-3
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            const int code = qv(rng);
            q[(size_t) r * qbytes + b / 4] |= (uint8_t) (code << (2 * (b % 4)));
            uint8_t* dn = &wn[(size_t) r * rb_new + (size_t) b * cb];
            uint8_t* dd = &wo[(size_t) r * rb_old + (size_t) b * (2 + cb)];
            for (size_t j = 0; j < cb; ++j) dn[j] = (uint8_t) byte(rng);
            std::memcpy(dd + 2, dn, cb);
            const uint16_t h = to_half(tq_detail::tq_half_to_float(cscale[b]) * sc.p2[code]);
            std::memcpy(dd, &h, 2);
        }
    sc.rowbase = rbase.data(); sc.colscale = cscale.data(); sc.q = q.data(); sc.qbytes = qbytes;
    std::vector<std::vector<uint8_t>> act(NT);
    std::vector<std::vector<float>> d(NT);
    std::vector<std::vector<int>> qq(NT);
    const void* ap[NT];
    for (int t = 0; t < NT; ++t) { act[t] = random_q8_0(n, rng, d[t], qq[t]); ap[t] = act[t].data(); }
    std::vector<float> o_new((size_t) NT * rows), o_old((size_t) NT * rows), o_isa((size_t) NT * rows);
    float* pn[NT]; float* po[NT]; float* pi[NT];
    for (int t = 0; t < NT; ++t) {
        pn[t] = o_new.data() + (size_t) t * rows; po[t] = o_old.data() + (size_t) t * rows; pi[t] = o_isa.data() + (size_t) t * rows;
    }
    tq_rows_r1q2(TqIsa::scalar, type, wn.data(), rb_new, n, ap, NT, pn, 0, rows / 3, sc);
    tq_rows_r1q2(TqIsa::scalar, type, wn.data(), rb_new, n, ap, NT, pn, rows / 3, rows, sc);
    tq_rows(TqIsa::scalar, type, wo.data(), rb_old, n, ap, NT, po, 0, rows);
    // relative to |want| + the outputs' RMS: a row whose blocks cancel to ~0 keeps the blocks' own error (fp16 of d,
    // 2^-11 a block), which is not small against the row's tiny result
    double ss = 0;
    for (int t = 0; t < NT; ++t)
        for (int r = 0; r < rows; ++r) {
            const double w = (double) po[t][r] * tq_detail::tq_half_to_float(rbase[r]);
            ss += w * w;
        }
    const double rms = std::sqrt(ss / (NT * rows)) + 1e-12;
    double worst = 0;
    int bad = 0;
    for (int t = 0; t < NT; ++t)
        for (int r = 0; r < rows; ++r) {
            const double want = (double) po[t][r] * tq_detail::tq_half_to_float(rbase[r]);
            const double got = pn[t][r];
            const double rel = std::fabs(got - want) / (std::fabs(want) + rms);
            worst = std::fmax(worst, rel);
            if (rel > 2e-3) ++bad;
        }
    std::printf("type %d n %d R1Q2 scalar vs the old layout (d = fp16 of the scale) x rowbase: %d of %d outside 2e-3 of "
                "|want| + RMS, worst %.2e\n", type, n, bad, NT * rows, worst);
    if (bad) fail("R1Q2 scalar differs from the old layout", type, n);
    for (TqIsa isa : {TqIsa::avx2, TqIsa::avx2vnni, TqIsa::avx2e, TqIsa::avx512}) {
        if (!tq_isa_ok(isa)) { std::printf("type %d R1Q2 %-6s not on this CPU: skipped\n", type, tq_isa_name(isa)); continue; }
        bool ok = true;
        for (int nt = 1; nt <= NT; ++nt) {
            tq_rows_r1q2(isa, type, wn.data(), rb_new, n, ap, nt, pi, 0, rows, sc);
            ok = ok && same_bits(o_isa.data(), o_new.data(), (size_t) nt * rows);
        }
        std::printf("type %d n %d R1Q2 %-6s vs the scalar R1Q2 kernel: %s\n", type, n, tq_isa_name(isa), ok ? "bitwise equal" : "DIFFERENT");
        if (!ok) fail("R1Q2 ISA differs from scalar", type, n);
    }
    // QuantLab (11/12): the bih3 lane-wise rows (STRATA_TQ_BIH3_LANES): the scalar kernel's sums to the float rounding
    if (tq_bih3_type(type) && tq_isa_ok(TqIsa::avx512)) {
        std::vector<tq_detail::TqAct> acts(NT);
        const tq_detail::TqAct* aptr[NT];
        for (int t = 0; t < NT; ++t) { tq_detail::unpack_act(ap[t], n, acts[t]); aptr[t] = &acts[t]; }
        double rms2 = 0;
        for (size_t i = 0; i < (size_t) NT * rows; ++i) rms2 += (double) o_new[i] * o_new[i];
        const double orms = std::sqrt(rms2 / (NT * rows)) + 1e-12;
        tq_detail::tq_rows_avx512_r1q2_bih3_lanes(K, wn.data(), rb_new, n, aptr, NT, pi, 0, rows, sc);
        double lw = 0;
        for (size_t i = 0; i < (size_t) NT * rows; ++i)
            lw = std::fmax(lw, std::fabs((double) o_isa[i] - o_new[i]) / (std::fabs((double) o_new[i]) + orms));
        std::printf("type %d n %d R1Q2 avx512 lane-wise vs the scalar R1Q2 kernel: worst %.2e of |want| + RMS  %s\n", type, n,
                    lw, lw < 1e-5 ? "ok" : "FAIL");
        if (!(lw < 1e-5)) fail("R1Q2 bih3 lane-wise rows differ", type, n);
    }
    // fused gate/up: the same rows as gate and up (own scales each: up uses rowbase reversed)
    std::vector<uint8_t> blob = wn;
    blob.insert(blob.end(), wn.begin(), wn.end());
    std::vector<uint16_t> rb_up(rbase.rbegin(), rbase.rend());
    tq_detail::TqScales su = sc;
    su.rowbase = rb_up.data();
    std::vector<float> f0((size_t) NT * rows), f1((size_t) NT * rows);
    float* q0[NT]; float* q1[NT];
    for (int t = 0; t < NT; ++t) { q0[t] = f0.data() + (size_t) t * rows; q1[t] = f1.data() + (size_t) t * rows; }
    tq_gu_rows_r1q2(TqIsa::scalar, type, blob.data(), rb_new, wn.size(), n, ap, NT, q0, 0, rows, sc, su);
    bool okg = true;
    for (int t = 0; t < NT && okg; ++t)
        for (int r = 0; r < rows; ++r) {
            const float g = pn[t][r], u = o_new[(size_t) t * rows + r] / tq_detail::tq_half_to_float(rbase[r]) *
                                              tq_detail::tq_half_to_float(rb_up[r]);
            const float want = (g / (1.f + std::exp(-g))) * u;
            if (std::fabs(q0[t][r] - want) > 1e-4f * (std::fabs(want) + 1e-3f)) { okg = false; break; }
        }
    for (TqIsa isa : {TqIsa::avx2, TqIsa::avx2vnni, TqIsa::avx2e, TqIsa::avx512}) {
        if (!tq_isa_ok(isa)) continue;
        tq_gu_rows_r1q2(isa, type, blob.data(), rb_new, wn.size(), n, ap, NT, q1, 0, rows, sc, su);
        okg = okg && same_bits(f0.data(), f1.data(), f0.size());
    }
    std::printf("type %d n %d R1Q2 gate/up rows (own scales, every ISA): %s\n", type, n, okg ? "ok" : "DIFFERENT");
    if (!okg) fail("R1Q2 gate/up rows differ", type, n);
}

// QuantLab (10/31): tqp_cpu_test --bench - the old rows against the R1Q2 rows on the same codes, every ISA this CPU
// has, ns per 128-value block and row (the decode's CPU experts: n 2560 gate/up rows, n 640 down rows)
void bench(int type, int n, int rows) {
    const int K = tq_kbits(type), nb = n / 128, qbytes = (nb + 3) / 4;
    std::mt19937 rng(7u + (uint32_t) type);
    std::uniform_int_distribution<int> byte(0, 255);
    const size_t cb = (size_t) 4 * K, rb_new = cb * nb, rb_old = (2 + cb) * nb;
    std::vector<uint8_t> wn(rb_new * rows), wo(rb_old * rows), q((size_t) qbytes * rows);
    for (auto& v : wn) v = (uint8_t) byte(rng);
    for (auto& v : q) v = (uint8_t) byte(rng);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t* dd = &wo[(size_t) r * rb_old + (size_t) b * (2 + cb)];
            const uint16_t h = 0x2000;
            std::memcpy(dd, &h, 2);
            std::memcpy(dd + 2, &wn[(size_t) r * rb_new + (size_t) b * cb], cb);
        }
    std::vector<uint16_t> rbase(rows, 0x3c00), cscale(nb, 0x2000);
    tq_detail::TqScales sc;
    sc.p2[0] = 0.84f; sc.p2[1] = 0.95f; sc.p2[2] = 1.07f; sc.p2[3] = 1.2f;
    sc.rowbase = rbase.data(); sc.colscale = cscale.data(); sc.q = q.data(); sc.qbytes = qbytes;
    std::vector<std::vector<uint8_t>> act(NT);
    std::vector<std::vector<float>> d(NT);
    std::vector<std::vector<int>> qq(NT);
    const void* ap[NT];
    for (int t = 0; t < NT; ++t) { act[t] = random_q8_0(n, rng, d[t], qq[t]); ap[t] = act[t].data(); }
    std::vector<float> o((size_t) NT * rows);
    float* po[NT];
    for (int t = 0; t < NT; ++t) po[t] = o.data() + (size_t) t * rows;
    auto time_ns = [&](auto&& fn) {
        fn();
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        int it = 0;
        double el = 0;
        do { fn(); ++it; el = std::chrono::duration<double>(clk::now() - t0).count(); } while (el < 0.4);
        return el * 1e9 / ((double) it * rows * nb);
    };
    for (TqIsa isa : {TqIsa::scalar, TqIsa::avx2, TqIsa::avx2vnni, TqIsa::avx2e, TqIsa::avx512}) {
        if (!tq_isa_ok(isa)) continue;
        for (int nt : {1, 2, 3}) {
            if (nt > NT) continue;
            const double t_old = time_ns([&] { tq_rows(isa, type, wo.data(), rb_old, n, ap, nt, po, 0, rows); });
            const double t_new = time_ns([&] { tq_rows_r1q2(isa, type, wn.data(), rb_new, n, ap, nt, po, 0, rows, sc); });
            std::printf("bench type %d n %4d %-6s nt %d: old %6.1f ns/block, R1Q2 %6.1f ns/block (%+.0f%%)\n", type, n,
                        tq_isa_name(isa), nt, t_old, t_new, 100.0 * (t_new / t_old - 1.0));
        }
    }
    if (tq_bih3_type(type) && tq_isa_ok(TqIsa::avx512)) {   // QuantLab (11/12): the lane-wise bih3 rows
        std::vector<tq_detail::TqAct> acts(NT);
        const tq_detail::TqAct* aptr[NT];
        for (int t = 0; t < NT; ++t) { tq_detail::unpack_act(ap[t], n, acts[t]); aptr[t] = &acts[t]; }
        for (int nt : {1, 2, 3}) {
            if (nt > NT) continue;
            const double t_e = time_ns([&] { tq_detail::tq_rows_avx512_r1q2_bih3(K, wn.data(), rb_new, n, aptr, nt, po, 0, rows, sc); });
            const double t_l = time_ns([&] { tq_detail::tq_rows_avx512_r1q2_bih3_lanes(K, wn.data(), rb_new, n, aptr, nt, po, 0, rows, sc); });
            std::printf("bench type %d n %4d avx512 bih3 rows alone nt %d: exact %6.1f, lane-wise %6.1f ns/block (%+.0f%%)\n",
                        type, n, nt, t_e, t_l, 100.0 * (t_l / t_e - 1.0));
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    // QuantLab (11/4) --lut <file>: everything with a pack's own codebook (tq_lut_f16.bin, strata/kernels/tq_lut.hpp)
    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--lut") {
            std::string err;
            if (!strata::kernels::tq_lut_load(args[i + 1], err)) {
                std::printf("--lut: %s  FAIL\n", err.c_str());
                return 1;
            }
            std::printf("tqp_cpu_test: codebook %s (sha256 %.16s, %s)\n", args[i + 1].c_str(),
                        strata::kernels::tq_lut_sha256().c_str(), strata::kernels::tq_lut_builtin() ? "the built-in one" : "not built-in");
            args.erase(args.begin() + (std::ptrdiff_t) i, args.begin() + (std::ptrdiff_t) i + 2);
            break;
        }
    if (!args.empty() && args[0] == "--bench") {
        for (int type : {145, 146, 170, 171, 186, 187, 190, 191})   // QuantLab (11/12): + bih3 K 6, 7, 10, 11
            for (int n : {2560, 640}) bench(type, n, n == 2560 ? 640 : 2560);
        return 0;
    }
    check_predicates();
    for (int type : {147, 148, 149}) check_type(type, true);
    for (int type : {144, 145, 146}) check_type(type, false);
    // QuantLab (10/31): R1Q2 blocks (K 6, 7, 8, 12, 14); (11/12) the bih3 types K 2, 3, 5, 8, 10, 13, 14
    for (int type : {145, 146, 144, 172, 174, 182, 183, 185, 188, 190, 193, 194})
        for (int n : {2560, 640}) check_r1q2(type, n);
    std::printf("tqp_cpu_test: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
