// tests/core/expert_layout_test.cpp - native_experts.txt (v2/v3/v4) and the expert arena filled from GGUF shards,
// without a model or a GPU.
//
//   1. Copies of the real packs' native_experts.txt (tests/data/native_experts: Q2_0, IQ2_XS, IQ3_XXS, IQ3_S,
//      Coder IQ1_M, Swift IQ2_XS; v2 and v3 headers) parse to the layout the pre-v4 reader produced: the test
//      parses each line again with that reader's rules (one shard name for all three roles) and compares every
//      field, and the totals with the header's.
//   2. The v4 shard column `gate,up,down` (an empty field = the --native shard); v5, four names, two names and a
//      trailing column are refused.
//   3. check_experts_gguf + load_experts_gguf on a synthetic arena at the real dimensions 2560/640 with Unsloth's
//      format pairs (Q4_K/Q5_1, Q5_K/Q8_0): layer 1's gate/up in another file than its down.  Every byte of the
//      arena is compared with the source tensors.  Then the refusals: an offset that is not the tensor's, a
//      truncated file, a role named in the wrong file, a type or a shape the pack does not say, a missing file.
#include "gguf_fixture.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using strata::kernels::cpu::ExpertLayout;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-expert-layout-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void write_text(const fs::path& p, const std::string& s) { std::ofstream(p, std::ios::binary) << s; }

bool load(const fs::path& dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    err.clear();
    return strata::kernels::cpu::expert_layout_load(dir.string(), n_layers, n_expert, err);
}

// ---- 1. the real packs' files
void real_packs(const fs::path& data) {
    for (const char* pack : {"q2_0", "iq2_xs", "iq3_xxs", "iq3_s", "coder-iq1_m", "swift-iq2_xs"}) {
        TempDir d;
        fs::copy_file(data / (std::string(pack) + ".txt"), d.path / "native_experts.txt");
        std::string err;
        const bool ok = load(d.path, 48, 512, err);
        check(ok, std::string(pack) + ": loads" + (ok ? "" : ": " + err));
        if (!ok) continue;
        const ExpertLayout& L = strata::kernels::cpu::expert_layout();
        // the pre-v4 reader's rules, applied to the same text
        std::ifstream in(d.path / "native_experts.txt");
        std::string line;
        long long hdr_n = -1, hdr_total = -1;
        bool same = L.native && L.n_layers == 48 && L.gguf_off.size() == 3 * 48;
        uint64_t max_blob = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                if (line.find("(n_expert ") == std::string::npos) continue;
                std::istringstream h(line.substr(line.find("(n_expert ") + 10));
                char comma = 0;
                std::string word;
                h >> hdr_n >> comma >> word >> hdr_total;   // "(n_expert N, total T; ..."
                continue;
            }
            std::istringstream ss(line);
            long long l = 0, gt = 0, dt = 0;
            unsigned long long off = 0, blob = 0, o[3] = {0, 0, 0};
            std::string file;
            ss >> l >> gt >> dt >> off >> blob >> o[0] >> o[1] >> o[2];
            ss >> file;
            same = same && L.fmt[(size_t) l].gu_type == gt && L.fmt[(size_t) l].d_type == dt &&
                   L.offset[(size_t) l] == off && L.bytes[(size_t) l] == blob;
            for (int r = 0; r < 3; ++r) {
                same = same && L.gguf_off[(size_t) (3 * l + r)] == o[r];
                const std::string got = L.gguf_file.empty() ? std::string() : L.gguf_file[(size_t) (3 * l + r)];
                same = same && got == file;
            }
            if (blob > max_blob) max_blob = blob;
        }
        same = same && L.n_expert == hdr_n && (long long) L.total == hdr_total && L.max_blob == max_blob;
        check(same, std::string(pack) + ": every field as the pre-v4 reader, totals as the header (v" +
                        std::to_string(L.version) + ", n_expert " + std::to_string(L.n_expert) + ")");
    }
}

// ---- 2. the v4 column and the refusals
std::string manifest(int version, const std::string& col1, int64_t blob, int n_expert = 2) {
    std::string s = "# strata native experts v" + std::to_string(version) +
                    ": layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down] "
                    "(n_expert " + std::to_string(n_expert) + ", total " + std::to_string(2 * blob * n_expert) + ")\n";
    s += "0 12 7 0 " + std::to_string(blob) + " 100 200 300\n";
    s += "1 12 7 " + std::to_string(blob * n_expert) + " " + std::to_string(blob) + " 400 500 600" +
         (col1.empty() ? "" : " " + col1) + "\n";
    return s;
}

void columns() {
    strata::kernels::cpu::NativeFmt f;
    std::string err;
    if (!strata::kernels::cpu::native_fmt(12, 7, 2560, 640, f, err)) {
        check(false, "Q4_K/Q5_1 native format: " + err);
        return;
    }
    const int64_t blob = (int64_t) f.bytes;
    TempDir d;
    write_text(d.path / "native_experts.txt", manifest(4, "B.gguf,B.gguf,", blob));
    bool ok = load(d.path, 2, 512, err);
    const ExpertLayout& L = strata::kernels::cpu::expert_layout();
    check(ok && L.version == 4 && L.gguf_file.size() == 6 && L.gguf_file[0].empty() && L.gguf_file[2].empty() &&
              L.gguf_file[3] == "B.gguf" && L.gguf_file[4] == "B.gguf" && L.gguf_file[5].empty() &&
              L.gguf_off[5] == 600,
          "v4: `B.gguf,B.gguf,` = gate and up in B.gguf, down in the --native shard" + (ok ? "" : ": " + err));
    write_text(d.path / "native_experts.txt", manifest(3, "A.gguf", blob));
    ok = load(d.path, 2, 512, err);
    check(ok && L.gguf_file[3] == "A.gguf" && L.gguf_file[4] == "A.gguf" && L.gguf_file[5] == "A.gguf",
          "v3: one name covers the three roles");
    write_text(d.path / "native_experts.txt", manifest(5, "", blob));
    ok = load(d.path, 2, 512, err);
    check(!ok && err.find("v5") != std::string::npos, "v5 is refused (\"" + err.substr(0, 40) + "...\")");
    write_text(d.path / "native_experts.txt", manifest(4, "a,b,c,d", blob));
    check(!load(d.path, 2, 512, err) && err.find("gate,up,down") != std::string::npos, "four names are refused");
    write_text(d.path / "native_experts.txt", manifest(4, "a,b", blob));
    check(!load(d.path, 2, 512, err), "two names are refused");
    write_text(d.path / "native_experts.txt", manifest(4, "a extra", blob));
    check(!load(d.path, 2, 512, err), "a column after the shard column is refused");
}

// ---- 3. the arena from synthetic shards
void arena() {
    using fixture::Tensor;
    constexpr int64_t H = 2560, FF = 640, NE = 2;
    const uint32_t Q4_K = 12, Q5_1 = 7, Q5_K = 13, Q8_0 = 8, Q3_K = 11, IQ3_S = 21;
    strata::kernels::cpu::NativeFmt f0, f1;
    std::string err;
    if (!strata::kernels::cpu::native_fmt(Q4_K, Q5_1, H, FF, f0, err) ||
        !strata::kernels::cpu::native_fmt(Q5_K, Q8_0, H, FF, f1, err)) {
        check(false, "Q4_K/Q5_1 and Q5_K/Q8_0 native formats: " + err);
        return;
    }
    TempDir d;
    const fs::path A = d.path / "M-00002-of-00003.gguf", B = d.path / "M-00003-of-00003.gguf";
    // A: layer 0 (all roles) and layer 1's down; B: layer 1's gate and up
    const std::vector<Tensor> ta = {{"blk.0.ffn_gate_exps.weight", {H, FF, NE}, Q4_K, 11},
                                    {"blk.0.ffn_up_exps.weight", {H, FF, NE}, Q4_K, 12},
                                    {"blk.0.ffn_down_exps.weight", {FF, H, NE}, Q5_1, 13},
                                    {"blk.1.ffn_down_exps.weight", {FF, H, NE}, Q8_0, 23}};
    const std::vector<Tensor> tb = {{"blk.1.ffn_gate_exps.weight", {H, FF, NE}, Q5_K, 21},
                                    {"blk.1.ffn_up_exps.weight", {H, FF, NE}, Q5_K, 22}};
    const auto wa = fixture::write(A, {}, ta);
    const auto wb = fixture::write(B, {}, tb);
    auto at = [](const fixture::Written& w, size_t i) { return std::to_string(w.data_start + w.offsets[i]); };
    const uint64_t total = (f0.bytes + f1.bytes) * NE;
    auto text = [&](const std::string& down1, const std::string& col1) {
        return "# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
               "[shard | gate,up,down] (n_expert 2, total " + std::to_string(total) + ")\n" +
               "0 12 7 0 " + std::to_string(f0.bytes) + " " + at(wa, 0) + " " + at(wa, 1) + " " + at(wa, 2) + "\n" +
               "1 13 8 " + std::to_string(f0.bytes * NE) + " " + std::to_string(f1.bytes) + " " + at(wb, 0) + " " +
               at(wb, 1) + " " + down1 + " " + col1 + "\n";
    };
    const fs::path pack = d.path / "pack";
    fs::create_directories(pack);
    write_text(pack / "native_experts.txt", text(at(wa, 3), B.filename().string() + "," + B.filename().string() + ","));
    bool ok = load(pack, 2, 512, err);
    check(ok, "the v4 manifest of the split layer loads" + (ok ? "" : ": " + err));
    if (!ok) return;
    const ExpertLayout& L = strata::kernels::cpu::expert_layout();
    ok = strata::core::check_experts_gguf(A.string(), L, err);
    check(ok, "check_experts_gguf: every span is its tensor" + (ok ? "" : ": " + err));
    std::vector<uint8_t> dst((size_t) L.total, 0xEE);
    const strata::core::LoadStats st = strata::core::load_experts_gguf(A.string(), dst.data(), L, 3);
    check(st.ok && st.bytes == L.total && L.total == total, "load_experts_gguf reads the whole arena (" +
                                                                 std::to_string(L.total) + " B)");
    // every byte: expert e's slice of role r comes from byte e * per[r] of that role's tensor
    bool exact = true;
    for (int64_t l = 0; l < 2 && exact; ++l) {
        const auto& fm = L.fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, fm.bytes - fm.down_off};
        const uint64_t off[3] = {0, fm.up_off, fm.down_off};
        const uint8_t seed[2][3] = {{11, 12, 13}, {21, 22, 23}};
        for (int64_t e = 0; e < NE && exact; ++e)
            for (int r = 0; r < 3 && exact; ++r) {
                const uint8_t* got = dst.data() + L.blob_offset(l, e) + off[r];
                for (uint64_t j = 0; j < per[r]; ++j)
                    if (got[j] != fixture::pattern(seed[l][r], (uint64_t) e * per[r] + j)) { exact = false; break; }
            }
    }
    check(exact, "every blob is [gate | up | down] of its expert, layer 1 from both files");

    {   // CS-T: the same blobs from FileExpertSource reading the shards in place (no experts.bin in the pack)
        strata::core::FileExpertSource fs;
        fs.set_gguf(A.string());
        std::string e1;
        const bool opened = fs.open(pack.string(), 2, NE, e1);
        check(opened && fs.gguf_mode(), "FileExpertSource opens the GGUF in place" + (opened ? "" : ": " + e1));
        if (opened) {
            bool same = true, copies = true, transient = true;
            std::vector<uint8_t> buf((size_t) std::max(f0.bytes, f1.bytes));
            for (int64_t l = 0; l < 2; ++l)
                for (int64_t e = 0; e < NE; ++e) {
                    fs.begin_layer(l, nullptr, 0);
                    const uint8_t* b = fs.blob(l, e);
                    const size_t n = (size_t) L.blob_bytes(l);
                    same = same && b != nullptr && std::memcmp(b, dst.data() + L.blob_offset(l, e), n) == 0;
                    copies = copies && fs.copy_blob(l, e, buf.data()) &&
                             std::memcmp(buf.data(), dst.data() + L.blob_offset(l, e), n) == 0;
                    const uint64_t released = fs.release(l, e);
#if defined(_WIN32)
                    const char* setting = std::getenv("STRATA_FILE_RELEASE");
                    const bool enabled = setting != nullptr && std::strcmp(setting, "1") == 0;
                    check(enabled ? released > 0 : released == 0, "GGUF mapped release honors its opt-in switch");
#else
                    check(released == 0, "GGUF mapped release remains unchanged outside Windows");
#endif
                    copies = copies && fs.copy_blob(l, e, buf.data()) &&
                             std::memcmp(buf.data(), dst.data() + L.blob_offset(l, e), n) == 0;
                    transient = transient && fs.transient(l, e);
                }
            check(same && copies && transient, "blob() and copy_blob() equal the loaded arena byte for byte (transient)");
            // a blob asked again in the same layer is the same buffer; its bytes hold through two more layers
            fs.begin_layer(0, nullptr, 0);
            const uint8_t* p = fs.blob(0, 1);
            const uint8_t* q = fs.blob(0, 1);
            fs.begin_layer(1, nullptr, 0);
            (void) fs.blob(1, 0);
            (void) fs.blob(1, 1);
            fs.begin_layer(0, nullptr, 0);
            (void) fs.blob(0, 0);
            check(p == q && std::memcmp(p, dst.data() + L.blob_offset(0, 1), (size_t) L.blob_bytes(0)) == 0,
                  "an assembled blob stays put while it is in use");
            check(fs.file_read_bytes() > 0, "the file tier counts its bytes (" + std::to_string(fs.file_read_bytes()) + ")");
            // prefetch: a layer's experts fetched together on several threads, then the same bytes from blob()
            bool pre = true;
            for (int round = 0; round < 3; ++round)
                for (int64_t l = 0; l < 2; ++l) {
                    fs.begin_layer(l, nullptr, 0);
                    fs.begin_layer(l == 0 ? 1 : 0, nullptr, 0);   // age the buffers so the prefetch refills them
                    fs.begin_layer(l, nullptr, 0);
                    const int64_t both[2] = {1, 0};
                    fs.set_fetch_threads(2);
                    fs.prefetch(l, both, 2);
                    for (int64_t e = 0; e < NE; ++e) {
                        const uint8_t* b = fs.blob(l, e);
                        pre = pre && b != nullptr &&
                              std::memcmp(b, dst.data() + L.blob_offset(l, e), (size_t) L.blob_bytes(l)) == 0;
                    }
                }
            check(pre && fs.file_blob_bytes() > 0 && fs.file_ms() >= 0.0,
                  "prefetch on 2 threads, then blob(): the same bytes");
        }
        fs.close();
        // with experts.bin present, the pack's file is mapped as before, whatever set_gguf says
        {
            std::ofstream eb(pack / "experts.bin", std::ios::binary);
            eb.write((const char*) dst.data(), (std::streamsize) dst.size());
        }
        strata::core::FileExpertSource fb;
        fb.set_gguf(A.string());
        std::string e2;
        const bool ob = fb.open(pack.string(), 2, NE, e2);
        const uint8_t* b11 = ob ? fb.blob(1, 1) : nullptr;
        check(ob && !fb.gguf_mode() && !fb.transient(1, 1) && b11 != nullptr &&
                  std::memcmp(b11, dst.data() + L.blob_offset(1, 1), (size_t) L.blob_bytes(1)) == 0,
              "with experts.bin the mapped file is used (unchanged behaviour)");
        fb.close();
        fs::remove(pack / "experts.bin");
    }

    auto refused = [&](const std::string& what, const std::string& needle) {
        std::string e2;
        const bool bad = load(pack, 2, 512, e2) && !strata::core::check_experts_gguf(A.string(),
                                                                                    strata::kernels::cpu::expert_layout(), e2);
        check(bad && e2.find(needle) != std::string::npos, what + " (\"..." + needle + "...\")");
    };
    const std::string both = B.filename().string() + "," + B.filename().string() + ",";
    write_text(pack / "native_experts.txt", text(std::to_string(std::stoull(at(wa, 3)) + 32), both));
    refused("an offset that is not the tensor's is refused", "starts at byte");
    write_text(pack / "native_experts.txt", text(at(wa, 3), ""));
    refused("layer 1's gate named in the file that does not hold it is refused", "is not in it");
    write_text(pack / "native_experts.txt", text(at(wa, 3), both + ",x"));
    std::string e3;
    check(!load(pack, 2, 512, e3), "a malformed column is refused at load");
    write_text(pack / "native_experts.txt", text(at(wa, 3), "missing.gguf,missing.gguf,"));
    refused("a shard that is not there is refused", "cannot open");
    fixture::write(B, {}, tb, /*cut=*/100);
    write_text(pack / "native_experts.txt", text(at(wa, 3), both));
    refused("a truncated shard is refused before anything is read", "past the end");

    // a type and a shape the pack does not say: Q3_K and IQ3_S are both 110 B per 256, so only the type differs
    TempDir d2;
    const fs::path C = d2.path / "C.gguf";
    const auto wc = fixture::write(C, {}, {{"blk.0.ffn_gate_exps.weight", {H, FF, NE}, IQ3_S, 31},
                                           {"blk.0.ffn_up_exps.weight", {H, FF, NE}, Q3_K, 32},
                                           {"blk.0.ffn_down_exps.weight", {FF, H, NE + 1}, Q5_1, 33}});
    strata::kernels::cpu::NativeFmt f2;
    if (!strata::kernels::cpu::native_fmt(Q3_K, Q5_1, H, FF, f2, err)) {
        check(false, "Q3_K/Q5_1 native format: " + err);
        return;
    }
    fs::create_directories(d2.path / "pack");
    write_text(d2.path / "pack" / "native_experts.txt",
               "# strata native experts v3: (n_expert 2, total " + std::to_string(f2.bytes * NE) + ")\n0 11 7 0 " +
                   std::to_string(f2.bytes) + " " + std::to_string(wc.data_start + wc.offsets[0]) + " " +
                   std::to_string(wc.data_start + wc.offsets[1]) + " " + std::to_string(wc.data_start + wc.offsets[2]) + "\n");
    ok = load(d2.path / "pack", 1, 512, err) && !strata::core::check_experts_gguf(C.string(),
                                                                                  strata::kernels::cpu::expert_layout(), err);
    check(ok && err.find("is IQ3_S") != std::string::npos, "a gate of another type than the pack's is refused");
    // the same bytes with the pack's types, but a down tensor of three experts where the pack says two
    fixture::write(C, {}, {{"blk.0.ffn_gate_exps.weight", {H, FF, NE}, Q3_K, 31},
                           {"blk.0.ffn_up_exps.weight", {H, FF, NE}, Q3_K, 32},
                           {"blk.0.ffn_down_exps.weight", {FF, H, NE + 1}, Q5_1, 33}});
    ok = load(d2.path / "pack", 1, 512, err) && !strata::core::check_experts_gguf(C.string(),
                                                                                  strata::kernels::cpu::expert_layout(), err);
    check(ok && err.find("is not [640, 2560, 2]") != std::string::npos, "a down tensor of another shape is refused");
}

// ---- 4. QuantLab v6: per-expert trellis formats (experts_k.txt)
void per_expert_k() {
    using strata::kernels::cpu::kTqPseudoBase;
    constexpr int64_t H = 2560, FF = 640, NE = 4;
    // layer 0 is Gyro-S's TQK6/TQK7 for all; layer 1 mixes K 5/6, 3/4 (pseudo/TQK4), 8/9 (TQ2_T/pseudo), 5/6
    const int gu[NE] = {150, kTqPseudoBase + 3, 144, 150}, dn[NE] = {145, 151, kTqPseudoBase + 9, 145};
    strata::kernels::cpu::NativeFmt f0, fe[NE];
    std::string err;
    bool ok = strata::kernels::cpu::native_fmt(145, 146, H, FF, f0, err);
    for (int e = 0; ok && e < NE; ++e) ok = strata::kernels::cpu::native_fmt(gu[e], dn[e], H, FF, fe[e], err);
    check(ok, "TQ formats for K 3..9 (SPEC ids and pseudo ids)" + (ok ? "" : ": " + err));
    if (!ok) return;
    // a trellis block is 2 + 4K bytes per 128 values: gate+up 2 x 640 x 2560, down 2560 x 640
    auto expect = [&](int kg, int kd) { return (uint64_t) 2 * (FF * H / 128) * (2 + 4 * kg) + (uint64_t) (H * FF / 128) * (2 + 4 * kd); };
    check(f0.bytes == 1049600 && fe[0].bytes == expect(5, 6) && fe[1].bytes == expect(3, 4) && fe[2].bytes == expect(8, 9),
          "expert bytes = 25600 x (3 + 4 Kg + 2 Kd)");
    uint64_t biggest = 0, sum1 = 0;
    for (int e = 0; e < NE; ++e) { biggest = std::max<uint64_t>(biggest, fe[e].bytes); sum1 += fe[e].bytes; }
    const uint64_t off1 = f0.bytes * NE;
    auto manifest6 = [&](int version, uint64_t row_blob, int row_gu, int row_d) {
        return "# strata native experts v" + std::to_string(version) + ": (n_expert " + std::to_string(NE) + ", total " +
               std::to_string(off1 + sum1) + ")\n0 145 146 0 " + std::to_string(f0.bytes) + " 100 200 300\n1 " +
               std::to_string(row_gu) + " " + std::to_string(row_d) + " " + std::to_string(off1) + " " +
               std::to_string(row_blob) + " 400 500 600\n";
    };
    std::string ks;
    for (int e = 0; e < NE; ++e) ks += "1 " + std::to_string(e) + " " + std::to_string(gu[e]) + " " + std::to_string(dn[e]) + "\n";
    const std::string had = "# strata hadamard v1: test\nblock 128\nlayer 0 1 1 1\nlayer 1 1 1 1\n";
    TempDir d;
    write_text(d.path / "native_experts.txt", manifest6(6, biggest, 144, kTqPseudoBase + 9));
    write_text(d.path / "experts_k.txt", "# layer expert gu_type d_type\n" + ks);
    write_text(d.path / "hadamard.txt", had);
    ok = load(d.path, 2, NE, err);
    check(ok, "v6 with experts_k.txt loads" + (ok ? "" : ": " + err));
    if (!ok) return;
    const ExpertLayout& L = strata::kernels::cpu::expert_layout();
    bool same = L.version == 6 && !L.layer_mixed(0) && L.layer_mixed(1) && L.cls_fmt[1].size() == 3;
    uint64_t at = off1;
    for (int e = 0; e < NE; ++e) {
        same = same && L.blob_offset(0, e) == (uint64_t) e * f0.bytes && L.expert_bytes(0, e) == f0.bytes &&
               L.fmt_of(0, e).gu_type == 145 && L.fmt_of(0, e).d_type == 146;
        same = same && L.blob_offset(1, e) == at && L.expert_bytes(1, e) == fe[e].bytes &&
               L.fmt_of(1, e).gu_type == gu[e] && L.fmt_of(1, e).d_type == dn[e] && L.fmt_of(1, e).bytes == fe[e].bytes;
        at += fe[e].bytes;
    }
    check(same, "layer 0 uniform; layer 1's offsets, sizes and formats per expert (3 classes)");
    check(L.layer_bytes(0) == off1 && L.layer_bytes(1) == sum1 && L.total == off1 + sum1 && L.max_blob == biggest &&
              L.blob_bytes(1) == biggest,
          "layer bytes, total, and the largest blob sizing the buffers");
    bool rot = true;
    for (int e = 0; e < NE; ++e)
        rot = rot && L.fmt_of(1, e).had_gu && L.fmt_of(1, e).had_d && L.fmt_of(1, e).had_block == 128;
    check(rot, "a mixed layer's formats rotate like the layer (hadamard.txt)");

    // the refusals
    std::filesystem::remove(d.path / "experts_k.txt");
    check(!load(d.path, 2, NE, err) && err.find("experts_k.txt is missing") != std::string::npos, "v6 without experts_k.txt is refused");
    write_text(d.path / "experts_k.txt", ks);
    write_text(d.path / "native_experts.txt", manifest6(5, biggest, 144, kTqPseudoBase + 9));
    check(!load(d.path, 2, NE, err) && err.find("is v5") != std::string::npos, "experts_k.txt beside a v5 manifest is refused");
    write_text(d.path / "native_experts.txt", manifest6(6, biggest, 144, kTqPseudoBase + 9));
    write_text(d.path / "experts_k.txt", ks.substr(0, ks.rfind("1 3 ")));
    check(!load(d.path, 2, NE, err) && err.find("lists 3 of 4") != std::string::npos, "a layer listing 3 of its 4 experts is refused");
    write_text(d.path / "experts_k.txt", ks);
    write_text(d.path / "native_experts.txt", manifest6(6, fe[0].bytes, 150, 145));
    check(!load(d.path, 2, NE, err) && err.find("largest expert") != std::string::npos,
          "a row that is not the layer's largest format is refused");
    write_text(d.path / "native_experts.txt", manifest6(6, biggest, 144, kTqPseudoBase + 9));
    write_text(d.path / "experts_k.txt", ks + "1 0 " + std::to_string(kTqPseudoBase + 13) + " 145\n");
    check(!load(d.path, 2, NE, err), "K 13 (no kernel) is refused");
}
}  // namespace

int main(int argc, char** argv) {
    // `expert_layout_test --real PACK_DIR NATIVE_SHARD`: a real pack's native_experts.txt against its model's files
    // (headers only - check_experts_gguf reads no expert byte)
    if (argc == 4 && std::string(argv[1]) == "--real") {
        std::string err;
        if (!load(argv[2], 48, 512, err)) { std::printf("layout refused: %s\n", err.c_str()); return 1; }
        const ExpertLayout& L = strata::kernels::cpu::expert_layout();
        const bool ok = strata::core::check_experts_gguf(argv[3], L, err);
        size_t split = 0;
        for (int64_t l = 0; l < L.n_layers && !L.gguf_file.empty(); ++l)
            split += L.gguf_file[(size_t) (3 * l)] != L.gguf_file[(size_t) (3 * l + 2)];
        std::printf("%s: v%d, %lld layers x %lld experts, %.2f GiB, %zu layer(s) split per role: %s%s\n", argv[2],
                    L.version, (long long) L.n_layers, (long long) L.n_expert, (double) L.total / (1u << 30), split,
                    ok ? "every span is its tensor" : "REFUSED: ", ok ? "" : err.c_str());
        return ok ? 0 : 1;
    }
    // `expert_layout_test --layout PACK_DIR`: load a pack's tables only (QuantLab: a v6 pack's per-expert formats)
    if (argc == 3 && std::string(argv[1]) == "--layout") {
        std::string err;
        if (!load(argv[2], 48, 512, err)) { std::printf("layout refused: %s\n", err.c_str()); return 1; }
        const ExpertLayout& L = strata::kernels::cpu::expert_layout();
        std::printf("%s: v%d, %lld layers x %lld experts, %.2f GiB (%llu B), largest blob %llu B\n", argv[2], L.version,
                    (long long) L.n_layers, (long long) L.n_expert, (double) L.total / (1u << 30),
                    (unsigned long long) L.total, (unsigned long long) L.max_blob);
        for (int64_t l = 0; l < L.n_layers; ++l) {
            if (!L.layer_mixed(l)) continue;
            std::printf("  layer %lld: mixed, %llu B at %llu, formats:", (long long) l, (unsigned long long) L.layer_bytes(l),
                        (unsigned long long) L.layer_offset(l));
            for (const auto& c : L.cls_fmt[(size_t) l]) {
                int n = 0;
                for (int64_t e = 0; e < L.n_expert; ++e) n += &L.fmt_of(l, e) == &c;
                std::printf(" %d/%d x%d", c.gu_type, c.d_type, n);
            }
            std::printf("; expert 511 at %llu (%llu B)\n", (unsigned long long) L.blob_offset(l, L.n_expert - 1),
                        (unsigned long long) L.expert_bytes(l, L.n_expert - 1));
        }
        return 0;
    }
    std::printf("expert_layout_test\n");
    if (argc < 2) {
        std::printf("usage: expert_layout_test <tests/data/native_experts>\n");
        return 2;
    }
    real_packs(argv[1]);
    columns();
    arena();
    per_expert_k();
    std::printf(g_fail ? "expert_layout_test: %d FAILED\n" : "expert_layout_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
