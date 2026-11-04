#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

// QuantLab (10/31), STRATA_QL_DENSE_TQ=<dir> (opt-in; unset: byte-identical): the projection matrices (attn_qkv, attn_gate,
// ssm_out, attn_q/k/v, attn_output, the shared experts) come from <dir>/dense_tq.bin as APR trellis blocks (types 144-146,
// 169-174, 2 + 4K bytes per 128 values) instead of the GGUF.  Their weights are stored Hadamard-rotated along the input
// (W' = W diag(s) H, per 128 values, signs s per input width from <dir>/dense_signs.txt), so a reader rotates its input
// the same way first (strata::kernels::hadamard_rows with ql_dense_tq_signs(width)).  Index <dir>/dense_tq.txt:
// "name type ne0 ne1 offset bytes" per line (ne0 = n_in).
bool ql_dense_tq_active();
const float* ql_dense_tq_signs(int width);   // device signs (loaded with the matrices); nullptr when absent
// the trellis bytes of `name` (n_in ne0, n_out ne1) when the folder names it: true with type/out filled; false with err
// empty when it does not, false with err set when it does but does not match
bool ql_dense_tq_read(const std::string& name, int64_t ne0, int64_t ne1, int& type, std::vector<uint8_t>& out,
                      std::string& err);
// QuantLab (11/4): dense matrices in other ORDINARY ggml types (Q4_K, Q5_K, Q5_1, ...: no trellis, no rotation), the
// GGUF's own block layout: index dense_nq.txt ("name type ne0 ne1 offset bytes", ne0 = n_in), bytes dense_nq.bin.
// From STRATA_QL_DENSE_NQ=<dir>, or else from the pack itself when it holds dense_nq.txt (STRATA_QL_DENSE_NQ=0: off).
// Replaced: the native projection matrices, output.weight, and the hyper-connection projections the STRATA_HC_Q8
// read takes from the GGUF (a Q5_1 one is dequantized and requantized to Q8_0 at load: the read stays Q8_0).
bool ql_dense_nq_init(const std::string& pack_dir, std::string& err);   // true with nothing on when there is none
bool ql_dense_nq_active();
const std::string& ql_dense_nq_dir();
// as ql_dense_tq_read: true with type/out filled when the index names `name` (and marks it used)
bool ql_dense_nq_read(const std::string& name, int64_t ne0, int64_t ne1, int& type, std::vector<uint8_t>& out,
                      std::string& err);
bool ql_dense_nq_has(const std::string& name);
std::string ql_dense_nq_unused();   // the names the index lists that nothing loaded ("" when none)
class WeightTable;
struct WeightRef;

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    /// With layer_hi >= 0, only the `blk.<l>.` matrices with layer_lo <= l < layer_hi are uploaded (a layer
    /// split's stage holds its own layers' projections, not the whole model's); the others keep data == nullptr.
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false, int64_t layer_lo = 0, int64_t layer_hi = -1);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// Layer split: the device bytes `load` would allocate for the matrices of layers [lb, le) - the same walk and
    /// filters, each matrix rounded up to the granule cudaMalloc backs it with - reading the GGUF headers and `table`
    /// only, so it allocates nothing and needs no device (#1238).  `table` must hold every tensor the shards carry.
    static bool weight_bytes_for(const std::vector<std::string>& shards, WeightTable& table, bool include_ple_key,
                                 int64_t lb, int64_t le, uint64_t& out, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    /// STRATA_HC_Q8=1: the hyper-connection down / up projections of the layers live only as Q8_0 (the bf16 copy is not
    /// loaded: `add_hc_skip` puts their rows in the arena's skip set, `load` quantizes them from the pack file).
    static bool hc_q8_requested();
    static bool add_hc_skip(const std::string& pack_dir, std::set<std::string>& skip);
    /// A projection's bf16 values for a path that has no Q8 kernel (the prompt path's GEMMs, the single-token read): the
    /// resident copy, or - when the row was skipped - the Q8_0 copy dequantized into a per-stream scratch (`slot` 0 / 1:
    /// two matrices alive at once).  nullptr if there is neither.
    /// `cls` 0 = the prompt path (a scratch pair shared by its streams, ordered by events: `hc_bf16_release` after the
    /// last GEMM that reads the slot), 1 = the layer / session read (its own pair: it is captured into graphs, where an
    /// event wait on an outside event is illegal). The scratches are allocated by `load`, never on first use.
    static const uint16_t* hc_bf16_view(const WeightRef& w, void* stream, int slot, int cls);
    static void hc_bf16_release(void* stream, int slot, int cls);
    uint64_t weight_bytes() const { return bytes_; }
    /// What the device lost to `load`: each matrix rounded up to the granule `cudaMalloc` backs it with.
    uint64_t allocated_bytes() const { return allocated_; }
    size_t tensor_count() const { return weights_.size() - packed_keys_.size(); }

private:
    std::vector<void*> weights_;
    std::vector<const void*> packed_keys_;   // GGUF-layout pointers registered with STRATA_Q8_PACKED=1
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    uint64_t allocated_ = 0;
};
} // namespace strata::core
