#include "strata/core/native_dense.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace strata::core {
namespace {
// ---- QuantLab (10/31): the trellis dense matrices (STRATA_QL_DENSE_TQ), see native_dense.hpp
struct TqEntry { int type; int64_t ne0, ne1; uint64_t offset, bytes; };
struct TqDense {
    bool on = false;
    std::string dir, error;
    std::map<std::string, TqEntry> index;
    std::map<int, std::vector<float>> signs;     // width -> host signs (kept: the device copies are cached by it)
    std::map<int, const float*> dsigns;          // width -> device signs, filled by NativeDense::load
};
TqDense& tq_dense() {
    static TqDense t = [] {
        TqDense d;
        const char* env = std::getenv("STRATA_QL_DENSE_TQ");
        if (env == nullptr || !*env) return d;
        d.dir = env;
        std::ifstream idx(d.dir + "/dense_tq.txt");
        if (!idx) { d.error = "STRATA_QL_DENSE_TQ: " + d.dir + "/dense_tq.txt is missing"; return d; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string name;
            TqEntry e{};
            if (!(ss >> name >> e.type >> e.ne0 >> e.ne1 >> e.offset >> e.bytes)) { d.error = "STRATA_QL_DENSE_TQ: bad line: " + line; return d; }
            d.index[name] = e;
        }
        std::ifstream sg(d.dir + "/dense_signs.txt");
        if (!sg) { d.error = "STRATA_QL_DENSE_TQ: " + d.dir + "/dense_signs.txt is missing"; return d; }
        while (std::getline(sg, line)) {
            std::istringstream ss(line);
            std::string key;
            if (!(ss >> key) || key != "signs") continue;
            int w = 0;
            ss >> w;
            std::vector<float> v;
            float x;
            while (ss >> x) v.push_back(x);
            if (w <= 0 || (int) v.size() != w) { d.error = "STRATA_QL_DENSE_TQ: bad signs line for width " + std::to_string(w); return d; }
            d.signs[w] = std::move(v);
        }
        d.on = !d.index.empty();
        return d;
    }();
    return t;
}
// `load` makes one cudaMalloc per projection matrix, and the driver backs each call with 2 MiB granules
// (measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and cost 2400).  So a stage's real
// footprint is the granule-rounded sum, not the payload sum.  This rule - a granule for every matrix of a
// MiB or more, the payload for the rest - prices the iq3_s pack's 300 matrices at 2255 MiB where they
// really take 2238 (+0.8%).  It over-prices, which is the safe side.  (Pricing every matrix at a granule is
// far worse, 2378 MiB: the driver packs the sub-MiB matrices into the slack of the rounded ones.)
constexpr uint64_t kAllocGranule = 2ull << 20;
constexpr uint64_t kSmallAlloc = 1ull << 20;
uint64_t alloc_bytes(uint64_t bytes) {
    return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule;
}
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_env() {
    // STRATA_HC_Q8: 1 = on (the GGUF's own Q8_0 projections, or - a bf16 pack - a Q8_0 copy quantized at load); 0 = off;
    // unset / auto = on for gfx12 (RDNA4: the R9700 decode +2.5 to +4%, KL 0.0003-0.0010 over its noise floor) unless
    // STRATA_GFX12_DEFAULTS=0; also on for gfx1151 (Strix Halo, STRATA_HC_Q8=0 turns it off), off elsewhere
    static const bool on = [] {
        const char* v = std::getenv("STRATA_HC_Q8");
        if (v != nullptr && v[0] == '1') return true;
        if (v != nullptr && v[0] != '\0' && std::strcmp(v, "auto") != 0) return false;
#if defined(STRATA_USE_HIP)
        {   // Strix Halo (gfx1151): on by default (decode +9.5%, json/code/tool/chat/agent +7.9..+10.7%); the answers differ in
            // near-tie tokens by about the size of the engine's own CPU-vs-GPU expert rounding (docs/AMD_HIP.md);
            // STRATA_HC_Q8=0 keeps the bf16 read
            int dev1151 = 0;
            cudaDeviceProp p1151;
            const bool g1151 = cudaGetDevice(&dev1151) == cudaSuccess && cudaGetDeviceProperties(&p1151, dev1151) == cudaSuccess &&
                               strata::kernels::gfx_arch_is_gfx1151(p1151.gcnArchName);
            (void) cudaGetLastError();
            if (g1151) {
                std::fprintf(stderr, "strata: Strix Halo (gfx1151): 8-bit hyper-connection weights are on (decode about 9%% faster, "
                                     "answers can differ in near-tie tokens); STRATA_HC_Q8=0 keeps the bf16 read\n");
                return true;
            }
        }
        const char* d = std::getenv("STRATA_GFX12_DEFAULTS");
        if (d != nullptr && d[0] == '0') return false;
        int dev = 0;
        cudaDeviceProp p;
        const bool g12 = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                         strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
        (void) cudaGetLastError();
        if (g12) std::fprintf(stderr, "strata: gfx12 (RDNA4): the hyper-connection projections are read as Q8_0 (quantized at load, "
                                      "decode faster, answers differ slightly); STRATA_HC_Q8=0 keeps the bf16 read\n");
        return g12;
#else
        return false;
#endif
    }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// QuantLab (11/9) STRATA_QL_HC_I8=1: the down / up Q8_0 copies repacked for the int8 read (fused_gr_q8_split); not with
// the Q8_0 inject rows (that read takes the bf16 ones)
bool ql_hc_i8_env() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_QL_HC_I8");
        if (v == nullptr || v[0] != '1') return false;
        const char* qi = std::getenv("STRATA_HC_Q8_INJECT");
        if (qi != nullptr && qi[0] == '1') {
            std::fprintf(stderr, "strata: STRATA_QL_HC_I8 is off: it does not take STRATA_HC_Q8_INJECT=1\n");
            return false;
        }
        return true;
    }();
    return on;
}
// repacks one down / up copy for the int8 read (the inject rows keep the Q8_0 layout)
bool ql_hc_i8_split(const std::string& name, void* p, uint64_t bytes, uint64_t& count, std::string& err) {
    if (!ql_hc_i8_env() || name.ends_with("_inject.weight")) return true;
    if (!strata::kernels::fused_gr_q8_split((uint8_t*) p, (int64_t) (bytes / 34), nullptr)) {
        err = "native dense (STRATA_QL_HC_I8): repacking " + name + " failed";
        return false;
    }
    strata::kernels::fused_gr_set_q8_i8(1);
    ++count;
    return true;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
// A bf16 matrix (the pack file's values) as Q8_0, ggml's reference rounding; the blocks are independent, so a few threads
std::vector<uint8_t> q8_0_of_bf16(const uint16_t* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    const uint64_t nb = n / 32;
    auto work = [&](uint64_t b0, uint64_t b1) {
        for (uint64_t b = b0; b < b1; ++b) {
            float v[32], amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const uint32_t u = (uint32_t) x[b * 32 + j] << 16;
                std::memcpy(&v[j], &u, 4);
                amax = (std::max)(amax, std::fabs(v[j]));
            }
            const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
            uint8_t* o = out.data() + b * 34;
            const uint16_t hb = half_bits(d);
            std::memcpy(o, &hb, 2);
            for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(v[j] * id);
        }
    };
    const unsigned nt = (std::min)(8u, (std::max)(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nt; ++i) pool.emplace_back(work, nb * i / nt, nb * (i + 1) / nt);
    for (auto& th : pool) th.join();
    return out;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
}

// ---- QuantLab (11/4): dense matrices in other ordinary ggml types (see native_dense.hpp)
namespace {
struct NqDense {
    bool on = false;
    std::string dir;
    std::map<std::string, TqEntry> index;
    std::set<std::string> used;
};
NqDense& nq_dense() {
    static NqDense d;
    return d;
}
float nq_half(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
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
// ggml's dequantize_row_q5_1: a block of 32 is {fp16 d, fp16 m, u32 qh, 16 bytes of nibbles}
std::vector<float> nq_q5_1_floats(const uint8_t* b, uint64_t n) {
    std::vector<float> y((size_t) n);
    for (uint64_t k = 0; k < n / 32; ++k) {
        const uint8_t* blk = b + k * 24;
        uint16_t dh, mh;
        uint32_t qh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&mh, blk + 2, 2);
        std::memcpy(&qh, blk + 4, 4);
        const float d = nq_half(dh), m = nq_half(mh);
        const uint8_t* qs = blk + 8;
        for (int j = 0; j < 16; ++j) {
            const uint32_t x0 = (qs[j] & 0x0Fu) | (((qh >> j) << 4) & 0x10u);
            const uint32_t x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10u);
            y[(size_t) (k * 32 + j)] = (float) x0 * d + m;
            y[(size_t) (k * 32 + j + 16)] = (float) x1 * d + m;
        }
    }
    return y;
}
}  // namespace

bool ql_dense_nq_init(const std::string& pack_dir, std::string& err) {
    NqDense& d = nq_dense();
    d = NqDense{};
    const char* env = std::getenv("STRATA_QL_DENSE_NQ");
    if (env != nullptr && (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0)) return true;
    std::string dir = env != nullptr && *env ? std::string(env) : pack_dir;
    if (dir.empty()) return true;
    std::ifstream idx(dir + "/dense_nq.txt");
    if (!idx) {
        if (env != nullptr && *env) { err = "STRATA_QL_DENSE_NQ: " + dir + "/dense_nq.txt is missing"; return false; }
        return true;   // a pack without dense replacements
    }
    uint64_t end = 0;
    std::string line;
    while (std::getline(idx, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name;
        TqEntry e{};
        if (!(ss >> name >> e.type >> e.ne0 >> e.ne1 >> e.offset >> e.bytes)) { err = "dense_nq.txt: bad line: " + line; return false; }
        if (strata::kernels::native_tq_type(e.type) || !strata::kernels::native_mmvq_supported(e.type) ||
            e.ne0 <= 0 || e.ne0 > INT_MAX || e.ne1 <= 0 || e.ne1 > INT_MAX ||
            e.bytes != strata::kernels::native_mmvq_weight_bytes(e.type, (int) e.ne0, (int) e.ne1)) {
            err = "dense_nq.txt: " + name + ": type " + std::to_string(e.type) + " " + std::to_string(e.ne0) + " x " +
                  std::to_string(e.ne1) + " is not an ordinary type the engine reads, or its byte count is wrong";
            return false;
        }
        if (!d.index.emplace(name, e).second) { err = "dense_nq.txt: duplicate " + name; return false; }
        end = (std::max)(end, e.offset + e.bytes);
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(dir + "/dense_nq.bin", ec);
    if (ec || size < end) { err = "dense_nq.bin in " + dir + " is missing or shorter than its index"; return false; }
    for (const auto& [name, e] : d.index)
        if (tq_dense().on && tq_dense().index.count(name)) {
            err = "dense_nq and STRATA_QL_DENSE_TQ both replace " + name; return false;
        }
    d.dir = dir;
    d.on = !d.index.empty();
    return true;
}
bool ql_dense_nq_active() { return nq_dense().on; }
const std::string& ql_dense_nq_dir() { return nq_dense().dir; }
bool ql_dense_nq_has(const std::string& name) { return nq_dense().on && nq_dense().index.count(name) > 0; }
bool ql_dense_nq_read(const std::string& name, int64_t ne0, int64_t ne1, int& type, std::vector<uint8_t>& out,
                      std::string& err) {
    NqDense& d = nq_dense();
    if (!d.on) return false;
    auto it = d.index.find(name);
    if (it == d.index.end()) return false;
    const TqEntry& e = it->second;
    if (e.ne0 != ne0 || e.ne1 != ne1) {
        err = "dense_nq: " + name + " is " + std::to_string(e.ne0) + " x " + std::to_string(e.ne1) + ", the model's " +
              std::to_string(ne0) + " x " + std::to_string(ne1);
        return false;
    }
    std::ifstream f(d.dir + "/dense_nq.bin", std::ios::binary);
    out.resize(e.bytes);
    f.seekg((std::streamoff) e.offset);
    if (!f || !f.read((char*) out.data(), (std::streamsize) e.bytes)) { err = "dense_nq.bin is short at " + name; return false; }
    type = e.type;
    d.used.insert(name);
    return true;
}
std::string ql_dense_nq_unused() {
    std::string out;
    for (const auto& [name, e] : nq_dense().index)
        if (!nq_dense().used.count(name)) out += (out.empty() ? "" : ", ") + name;
    return out;
}

bool ql_dense_tq_active() { return tq_dense().on && tq_dense().error.empty(); }
bool ql_dense_tq_read(const std::string& name, int64_t ne0, int64_t ne1, int& type, std::vector<uint8_t>& out,
                      std::string& err) {
    auto& t = tq_dense();
    if (!t.on) return false;
    auto it = t.index.find(name);
    if (it == t.index.end()) return false;
    const TqEntry& e = it->second;
    if (!strata::kernels::native_tq_type(e.type) || e.ne0 != ne0 || e.ne1 != ne1 ||
        e.bytes != strata::kernels::native_mmvq_weight_bytes(e.type, (int) ne0, (int) ne1)) {
        err = "STRATA_QL_DENSE_TQ: " + name + " does not match its expected shape or type";
        return false;
    }
    std::ifstream f(t.dir + "/dense_tq.bin", std::ios::binary);
    out.resize(e.bytes);
    f.seekg((std::streamoff) e.offset);
    if (!f || !f.read((char*) out.data(), (std::streamsize) e.bytes)) {
        err = "STRATA_QL_DENSE_TQ: dense_tq.bin is short at " + name;
        return false;
    }
    type = e.type;
    return true;
}
const float* ql_dense_tq_signs(int width) {
    auto& t = tq_dense();
    auto it = t.dsigns.find(width);
    return it == t.dsigns.end() ? nullptr : it->second;
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
// The byte walk `load` does for the layers [lb, le), without the allocations: the same filters in the same order,
// reading the GGUF headers and the canonical table only, so it needs no device.  A later stage's projections are not
// on the card when the split search prices that stage, and `load` runs one cudaMalloc per matrix (#1238).  The two
// have to stay in step.  A repeated name is skipped rather than refused (`load` refuses it).
bool NativeDense::weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                   int64_t lb, int64_t le, uint64_t& out, std::string& err) {
    out = 0;
    try {
        std::set<std::string> seen;
        uint64_t total = 0;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key)) continue;
                if (tensor.name.rfind("blk.", 0) == 0) {
                    const long l = std::strtol(tensor.name.c_str() + 4, nullptr, 10);
                    if (l < lb || l >= le) continue;
                }
                if (!seen.insert(tensor.name).second) continue;
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                const auto& ref = found->second;
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                const auto tq = tq_dense().index.find(tensor.name);   // QuantLab: the trellis matrix replaces it
                const auto nq = nq_dense().index.find(tensor.name);   // QuantLab (11/4): or an ordinary type does
                total += alloc_bytes(tq_dense().on && tq != tq_dense().index.end()
                                         ? tq->second.bytes
                                         : nq_dense().on && nq != nq_dense().index.end()
                                               ? nq->second.bytes
                                               : strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1));
            }
        }
        out = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

// STRATA_HC_Q8 without a bf16 copy: the two dequantization scratches (one bf16 matrix each) of `hc_bf16_view`, per device,
// allocated by `load` - never on first use, so a graph capture cannot meet the allocation
static void* g_hc_scratch[64][2][2] = {};     // [device][class][slot]
static cudaEvent_t g_hc_ev[64][2] = {};        // [device][slot]: the prompt path's last reader of the slot
static bool g_hc_ev_set[64][2] = {};

NativeDense::~NativeDense() {
    for (auto& dev : g_hc_scratch)
        for (auto& cl : dev)
            for (void*& b : cl)
                if (b != nullptr && std::find(weights_.begin(), weights_.end(), b) != weights_.end()) b = nullptr;
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    if (!tq_dense().error.empty()) { err = tq_dense().error; return false; }
    std::ifstream tq_bin;
    uint64_t tq_count = 0, tq_bytes = 0, tq_replaced = 0;
    uint64_t nq_count = 0, nq_bytes = 0, nq_replaced = 0;   // QuantLab (11/4): dense_nq
    if (tq_dense().on) {
        tq_bin.open(tq_dense().dir + "/dense_tq.bin", std::ios::binary);
        if (!tq_bin) { err = "STRATA_QL_DENSE_TQ: " + tq_dense().dir + "/dense_tq.bin is missing"; return false; }
        for (auto& [w, host] : tq_dense().signs) {
            const float* d = strata::kernels::hadamard_device_signs(host.data(), w);
            if (d == nullptr) { err = "STRATA_QL_DENSE_TQ: the signs of width " + std::to_string(w) + " could not be uploaded"; return false; }
            tq_dense().dsigns[w] = d;
        }
    }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0, allocated = 0;
        uint64_t ql_i8_count = 0;   // QuantLab STRATA_QL_HC_I8: down / up copies repacked
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {
                // splitter may copy general.architecture into every shard (Huihui abliterated re-split)
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                // QuantLab (10/31): a matrix STRATA_QL_DENSE_TQ names is uploaded from dense_tq.bin in its trellis type
                const auto tqe = tq_dense().on ? tq_dense().index.find(tensor.name) : tq_dense().index.end();
                const bool tq_this = tq_dense().on && tqe != tq_dense().index.end();
                std::vector<uint8_t> tq_host;
                if (tq_this) {
                    const TqEntry& e = tqe->second;
                    if (!strata::kernels::native_tq_type(e.type) || e.ne0 != ref.ne0 || e.ne1 != ref.ne1 ||
                        e.bytes != strata::kernels::native_mmvq_weight_bytes(e.type, (int) ref.ne0, (int) ref.ne1) ||
                        ql_dense_tq_signs((int) ref.ne0) == nullptr) {
                        err = "STRATA_QL_DENSE_TQ: " + tensor.name + " does not match (type " + std::to_string(e.type) +
                              ", " + std::to_string(e.ne0) + " x " + std::to_string(e.ne1) + ", or no signs for its width)";
                        return false;
                    }
                    tq_host.resize(e.bytes);
                    tq_bin.seekg((std::streamoff) e.offset);
                    if (!tq_bin.read((char*) tq_host.data(), (std::streamsize) e.bytes)) {
                        err = "STRATA_QL_DENSE_TQ: dense_tq.bin is short at " + tensor.name; return false;
                    }
                    ++tq_count;
                    tq_bytes += e.bytes;
                    tq_replaced += strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1);
                }
                // QuantLab (11/4): or dense_nq replaces it in another ordinary type (no rotation)
                int nq_type = -1;
                std::vector<uint8_t> nq_host;
                const bool nq_this = !tq_this && ql_dense_nq_read(tensor.name, ref.ne0, ref.ne1, nq_type, nq_host, err);
                if (!err.empty()) return false;
                if (nq_this) {
                    ++nq_count;
                    nq_bytes += nq_host.size();
                    nq_replaced += strata::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1);
                }
                const int up_type = tq_this ? tqe->second.type : nq_this ? nq_type : (int) tensor.type;
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(up_type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), tq_this ? (const void*) tq_host.data()
                                                    : nq_this ? (const void*) nq_host.data() : gguf.tensor_data(tensor), bytes,
                                        cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                allocated += alloc_bytes(bytes);
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (!tq_this && !nq_this && tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = cudaMalloc(&packed_allocation, bytes);
                    packed.reset(packed_allocation);
                    if (packed_status == cudaSuccess)
                        packed_status = cudaMemcpy(packed.get(), host.data(), bytes, cudaMemcpyHostToDevice);
                    if (packed_status != cudaSuccess) {
                        err = "native dense packed upload " + tensor.name + ": " + cudaGetErrorString(packed_status);
                        return false;
                    }
                }
                pending.push_back(Pending{&ref, up_type, bytes, std::move(data), std::move(packed)});
            }
            if (hc_q8_env())
                for (const auto& tensor : gguf.tensors()) {
                    // S25: the F32 inject rows hold BF16-exact values: the read takes the pack's BF16 rows unless
                    // STRATA_HC_Q8_INJECT=1 asks for a Q8_0 copy
                    static const bool q8_inject = [] { const char* v = std::getenv("STRATA_HC_Q8_INJECT"); return v && v[0] == '1'; }();
                    const bool f32_inject = q8_inject && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type != 8 && !f32_inject) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2) continue;
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    void* p = nullptr;
                    std::vector<uint8_t> q8;
                    if (f32_inject) q8 = q8_0_of((const float*) gguf.tensor_data(tensor), (uint64_t) ref.ne0 * (uint64_t) ref.ne1);
                    // QuantLab (11/4): dense_nq's copy of this projection (Q8_0 as it is; Q5_1 dequantized, then Q8_0)
                    int hq_type = -1;
                    std::vector<uint8_t> hq;
                    const bool hq_this = !f32_inject && ql_dense_nq_read(tensor.name, ref.ne0, ref.ne1, hq_type, hq, err);
                    if (!err.empty()) return false;
                    if (hq_this) {
                        const uint64_t n = (uint64_t) ref.ne0 * (uint64_t) ref.ne1;
                        if (hq_type == 8) q8 = std::move(hq);
                        else if (hq_type == 7) { const std::vector<float> fv = nq_q5_1_floats(hq.data(), n); q8 = q8_0_of(fv.data(), n); }
                        else { err = "dense_nq: " + tensor.name + ": a hyper-connection projection must be Q8_0 or Q5_1"; return false; }
                        ++nq_count;
                    }
                    if (cudaMalloc(&p, bytes) != cudaSuccess ||
                        cudaMemcpy(p, f32_inject || hq_this ? (const void*) q8.data() : gguf.tensor_data(tensor), bytes,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
                    }
                    ref.hc_q8 = p;
                    weights_.push_back(p);
                    hc_q8_bytes += bytes;
                    if (!ql_hc_i8_split(tensor.name, p, bytes, ql_i8_count, err)) return false;
                }
        }
        if (hc_q8_env()) {
            // a bf16 pack stores no Q8_0 copy: quantize the layers' down / up projections. A row the arena kept
            // (--keep-canonical, a layer split) is quantized on the GPU from the arena's bf16; a skipped row (the
            // single-GPU default: add_hc_skip) from the pack file on the host, and has no bf16 copy at all. The
            // inject rows stay bf16 (82 KB), the final mixer stays bf16 (it runs once a window).
            uint64_t made = 0;
            for (auto& [name, ref] : table.table_) {
                if (ref.hc_q8 != nullptr || name.rfind("blk.", 0) != 0 || !in_range(name) ||
                    !hc_q8_name(name) || !(name.ends_with("_down.weight") || name.ends_with("_up.weight"))) continue;
                const bool skipped = ref.data == nullptr;
                if (skipped && ref.resident) continue;
                if (ref.ne0 % 32 != 0 || ref.elements != ref.ne0 * ref.ne1 || (!skipped && ref.bytes != (uint64_t) ref.elements * 2)) {
                    err = "native dense (STRATA_HC_Q8): " + name + " is not a bf16 matrix"; return false;
                }
                const uint64_t nb = (uint64_t) ref.elements / 32, bytes = nb * 34;
                void* p = nullptr;
                if (cudaMalloc(&p, bytes) != cudaSuccess) { err = "native dense (STRATA_HC_Q8) alloc " + name; return false; }
                weights_.push_back(p);
                if (skipped) {
                    std::vector<uint16_t> host;
                    if (!table.read_bf16_host(name, host, err)) return false;
                    const std::vector<uint8_t> q8 = q8_0_of_bf16(host.data(), (uint64_t) host.size());
                    if (cudaMemcpy(p, q8.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8) upload " + name; return false;
                    }
                } else {
                    strata::kernels::fused_gr_quantize_q8_0((const uint16_t*) ref.data, (uint8_t*) p, (int64_t) nb, nullptr);
                }
                if (!ql_hc_i8_split(name, p, bytes, ql_i8_count, err)) return false;   // (after the quantization, same stream)
                ref.hc_q8 = p;
                hc_q8_bytes += bytes;
                allocated += alloc_bytes(bytes);
                ++made;
            }
            if (made && (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess)) {
                err = "native dense (STRATA_HC_Q8): quantizing the projections failed"; return false;
            }
        }
        if (hc_q8_env()) {   // the scratches of hc_bf16_view (rows without a bf16 copy), here and not on first use
            uint64_t elems = 0;
            for (const auto& [name, ref] : table.table_)
                if (ref.hc_q8 != nullptr && ref.data == nullptr) elems = (std::max)(elems, (uint64_t) ref.elements);
            int dev = 0;
            if (elems != 0 && cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && g_hc_scratch[dev][0][0] == nullptr) {
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < 2; ++i) {
                        void* b = nullptr;
                        if (cudaMalloc(&b, (size_t) elems * 2) != cudaSuccess) { err = "native dense (STRATA_HC_Q8): scratch alloc"; return false; }
                        weights_.push_back(b);
                        g_hc_scratch[dev][c][i] = b;
                    }
                for (int i = 0; i < 2; ++i)
                    if (g_hc_ev[dev][i] == nullptr && cudaEventCreateWithFlags(&g_hc_ev[dev][i], cudaEventDisableTiming) != cudaSuccess) {
                        err = "native dense (STRATA_HC_Q8): scratch event"; return false;
                    }
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            if (item.packed) {
                strata::kernels::native_q8_0_packed_register(item.data.get(), item.packed.get(), (int) item.ref->ne0,
                                                             (int) item.ref->ne1);
                packed_keys_.push_back(item.data.get());
                packed_bytes += item.bytes;
                ++packed_count;
                weights_.push_back(item.packed.release());
            }
            weights_.push_back(item.data.release());
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        if (tq_count)
            std::fprintf(stderr, "strata: QuantLab: STRATA_QL_DENSE_TQ: %llu projection matrices as trellis blocks from %s "
                                 "(%.1f MiB in place of %.1f MiB)\n", (unsigned long long) tq_count, tq_dense().dir.c_str(),
                         tq_bytes / 1048576.0, tq_replaced / 1048576.0);
        if (nq_count)
            std::fprintf(stderr, "strata: QuantLab: dense_nq: %llu matrices from %s (projections %.1f MiB in place of "
                                 "%.1f MiB; hyper-connection ones requantized to Q8_0 for the STRATA_HC_Q8 read)\n",
                         (unsigned long long) nq_count, ql_dense_nq_dir().c_str(), nq_bytes / 1048576.0,
                         nq_replaced / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        allocated_ = allocated;
        if (hc_q8_env())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        if (ql_i8_count > 0)
            std::fprintf(stderr, "strata: STRATA_QL_HC_I8=1: %llu down / up projections repacked for the int8 hyper-connection read\n",
                         (unsigned long long) ql_i8_count);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
bool NativeDense::hc_q8_requested() { return hc_q8_env(); }

bool NativeDense::add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip) {
    std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb");
    if (!f) return false;
    char line[1024], name[256];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
        const std::string n = name;
        if (n.rfind("blk.", 0) == 0 && hc_q8_name(n) && (n.ends_with("_down.weight") || n.ends_with("_up.weight"))) skip.insert(n);
    }
    std::fclose(f);
    return true;
}

const uint16_t* NativeDense::hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls) {
    if (w.data != nullptr) return (const uint16_t*) w.data;
    if (w.hc_q8 == nullptr) return nullptr;
    // the device's scratch (allocated at load, one pair per device): the prompt path and the single-token read run one
    // after the other on a device, and each use is stream-ordered after the previous GEMM that read the slot
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || slot < 0 || slot > 1 || cls < 0 || cls > 1) return nullptr;
    void* buf = g_hc_scratch[dev][cls][slot];
    if (buf == nullptr) return nullptr;
    // the prompt path: the scratch may still be read by a GEMM another stream enqueued (batch serving: several slots' prompts)
    if (cls == 0 && g_hc_ev_set[dev][slot]) cudaStreamWaitEvent((cudaStream_t) stream, g_hc_ev[dev][slot], 0);
    strata::kernels::fused_gr_dequant_q8_0((const uint8_t*) w.hc_q8, (uint16_t*) buf, w.elements / 32, stream);
    return (const uint16_t*) buf;
}

void NativeDense::hc_bf16_release(void* stream, int slot, int cls) {
    int dev = 0;
    if (cls != 0 || slot < 0 || slot > 1 || cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64 || g_hc_ev[dev][slot] == nullptr) return;
    cudaEventRecord(g_hc_ev[dev][slot], (cudaStream_t) stream);
    g_hc_ev_set[dev][slot] = true;
}
} // namespace strata::core
