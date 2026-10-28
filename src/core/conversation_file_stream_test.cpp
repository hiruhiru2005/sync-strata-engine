// QuantLab (10/29): CPU-only tests for the streamed session files (STRATA_QL_SESSION_STREAM; conversation_file.hpp):
// the two-pass reader (SessionFileStream), the bounded-RAM writer (session_file_write_streamed) and their RAM estimates.
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.  The format is the one conversation_file_test
// freezes: the same sample gives the same golden bytes here.
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// the sample of conversation_file_test (its golden file: kGoldenSize bytes, kGoldenHash)
ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}
std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}
ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(int32_t(1000 + i));
    c.imgs = {{3, 0x1234567890abcdefull ^ seed}};
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}
SavedConversation sample() {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.cvec = false;
    s.live = checkpoint(1000, 1);
    s.checkpoints.push_back(checkpoint(10, 2));
    s.checkpoints.push_back(checkpoint(500, 3));
    for (int layer = 0; layer < 3; ++layer) {
        ConversationKv kv;
        kv.format = 2 + layer; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256;
        kv.page_size = 64; kv.pooled_rows = layer == 2 ? 0 : 9; kv.idx_dim = 128;
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(layer));
        kv.v = pattern(4096 + layer, uint8_t(layer + 10));
        kv.k_scale = pattern(layer == 1 ? 0 : 512, uint8_t(layer + 20));
        kv.v_scale = pattern(512, uint8_t(layer + 30));
        kv.pooled = pattern(layer == 2 ? 0 : 9 * 128 * 4, uint8_t(layer + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}
constexpr size_t kGoldenSize = 17878535;
constexpr uint64_t kGoldenHash = 0x70f812b350360d09ull;

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return true;
}
std::vector<char> slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const fs::path& p, const std::vector<char>& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize) d.size());
}
bool no_temp(const fs::path& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) return false;
    }
    return true;
}

std::vector<uint8_t>& state_part(ConversationCheckpoint& c, size_t part) {
    switch (part) {
        case 0: return c.gdn;
        case 1: return c.ple;
        case 2: return c.tails;
        case 3: return c.dead;
        default: return c.block_pos;
    }
}
ConversationBuffer& kv_part(ConversationKv& k, size_t part) {
    switch (part) {
        case 0: return k.k;
        case 1: return k.v;
        case 2: return k.k_scale;
        case 3: return k.v_scale;
        default: return k.pooled;
    }
}
const SessionFileArray& located(const SessionFileLayout& f, const SessionStreamPiece& p) {
    if (p.kind == 0) return f.live.state[p.part];
    if (p.kind == 1) return f.checkpoints[p.index].state[p.part];
    return f.kv[p.index].parts[p.part];
}

// What a streamed restore sees, rebuilt in RAM: pass 1's layout gives the shapes, pass 2's pieces the bytes.
struct Streamed {
    SavedConversation image;
    SessionFileLayout layout;
    size_t bytes = 0, pieces = 0, largest_piece = 0;
    bool in_file_order = true, whole_arrays = true;
};
bool stream_read(const fs::path& p, const SessionFileIdentity& id, Streamed& out, std::string& error,
                 const SessionReadLimits& limits = {}, const SessionStreamOptions& opt = {}, SessionStatus* st = nullptr) {
    SessionFileStream file;
    Streamed s;
    if (!file.verify(p.string(), id, limits, opt, s.layout, s.bytes, error, st)) return false;
    const SessionFileLayout& f = s.layout;
    SavedConversation& img = s.image;
    img.geometry = f.geometry; img.layer_lo = f.layer_lo; img.layer_hi = f.layer_hi; img.cvec = f.cvec;
    auto shell = [](const SessionFileCheckpoint& from, ConversationCheckpoint& c) {
        c.ids = from.ids; c.imgs = from.imgs; c.used = from.used;
        for (size_t part = 0; part < 5; ++part) state_part(c, part).assign((size_t) from.state[part].bytes, 0xEE);
    };
    shell(f.live, img.live);
    img.checkpoints.resize(f.checkpoints.size());
    for (size_t i = 0; i < f.checkpoints.size(); ++i) shell(f.checkpoints[i], img.checkpoints[i]);
    img.kv.resize(f.kv.size());
    for (size_t j = 0; j < f.kv.size(); ++j) {
        const SessionFileKv& from = f.kv[j];
        ConversationKv& k = img.kv[j];
        k.format = from.format; k.cells = from.cells; k.heads = from.heads; k.head_dim = from.head_dim;
        k.page_size = from.page_size; k.pooled_rows = from.pooled_rows; k.idx_dim = from.idx_dim;
        for (size_t part = 0; part < 5; ++part) kv_part(k, part).resize((size_t) from.parts[part].bytes, 0xEE);
    }
    uint64_t last = 0;   // the file offset just past the previous piece
    SessionStreamPiece open{-1, 0, 0, 0};
    uint64_t open_done = 0;
    const SessionStreamSink sink = [&](const SessionStreamPiece& piece, const uint8_t* data, size_t n) -> bool {
        const SessionFileArray& a = located(f, piece);
        const uint64_t at = a.offset + piece.at;
        if (at < last) s.in_file_order = false;
        last = at + n;
        // each array from its start to its end: a new array starts at 0, an open one goes on where it stopped
        const bool same_array = open.kind == piece.kind && open.index == piece.index && open.part == piece.part;
        if (same_array ? piece.at != open_done : piece.at != 0) s.whole_arrays = false;
        open = piece;
        open_done = piece.at + n;
        ++s.pieces;
        s.largest_piece = std::max(s.largest_piece, n);
        if (piece.kind == 0 || piece.kind == 1) {
            ConversationCheckpoint& c = piece.kind == 0 ? img.live : img.checkpoints.at(piece.index);
            std::vector<uint8_t>& v = state_part(c, piece.part);
            if (piece.at + n > v.size()) { error = "piece out of range"; return false; }
            std::memcpy(v.data() + piece.at, data, n);
            return true;
        }
        ConversationBuffer& b = kv_part(img.kv.at(piece.index), piece.part);
        return b.visit((size_t) piece.at, n, [&](uint8_t* d, size_t c, size_t where) {
            std::memcpy(d, data + (where - (size_t) piece.at), c);
            return true;
        });
    };
    if (!file.deliver(sink, error, st)) return false;
    out = std::move(s);
    return true;
}

// the sample's K/V layers as streamed sources (the file bytes are those of the captured image)
std::vector<SessionKvSource> kv_sources(const SavedConversation& s) {
    std::vector<SessionKvSource> sources;
    for (const auto& k : s.kv) {
        SessionKvSource src;
        src.format = k.format; src.cells = k.cells; src.heads = k.heads; src.head_dim = k.head_dim;
        src.page_size = k.page_size; src.pooled_rows = k.pooled_rows; src.idx_dim = k.idx_dim;
        const std::array<const ConversationBuffer*, 5> parts = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t i = 0; i < 5; ++i) src.sizes[i] = parts[i]->size();
        src.read = [parts](size_t part, size_t offset, void* dst, size_t n) { return parts[part]->read(dst, offset, n); };
        sources.push_back(std::move(src));
    }
    return sources;
}
// the live GDN rows as a source (the engine reads them from the device)
SessionKvSource gdn_source(const std::vector<uint8_t>& rows) {
    SessionKvSource src;
    src.sizes = {rows.size(), 0, 0, 0, 0};
    const std::vector<uint8_t>* p = &rows;
    src.read = [p](size_t part, size_t offset, void* dst, size_t n) {
        if (part != 0 || offset > p->size() || n > p->size() - offset) return false;
        std::memcpy(dst, p->data() + offset, n);
        return true;
    };
    return src;
}

SessionReadLimits exact(const SavedConversation& s) {
    SessionReadLimits l;
    l.max_tokens = 1000; l.max_checkpoints = 2; l.max_kv_layers = 3;
    l.geometry = s.geometry;
    l.layer_range = std::make_pair(s.layer_lo, s.layer_hi);
    l.max_state_bytes = {4099, 77, 301, 64, 8};
    for (const auto& k : s.kv) l.max_kv_bytes.push_back({k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()});
    return l;
}
} // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata-session-stream-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    const SavedConversation original = sample();
    const fs::path good = dir / "good.bin";
    size_t written = 0;
    std::string error;

    // the switch is read from the environment once
    {
        const char* v = std::getenv("STRATA_QL_SESSION_STREAM");
        check(session_stream_requested() == (v != nullptr && std::atoi(v) != 0), "the switch follows the environment");
    }
    // the ordinary writer is unchanged: the golden bytes of format v1
    check(session_file_write(good.string(), original, id, written, error), "ordinary write");
    const std::vector<char> image = slurp(good);
    check(image.size() == kGoldenSize && session_hash64(image.data(), image.size(), 0) == kGoldenHash,
          "the ordinary writer still gives format v1's golden bytes");

    // the two-pass reader rebuilds the same image, every array whole and in file order, at most a block a piece
    {
        Streamed s;
        check(stream_read(good, id, s, error), "stream: verify and deliver");
        check(s.bytes == image.size() && s.layout.file_bytes == image.size(), "stream: the file size");
        check(same(s.image, original), "stream: the delivered bytes rebuild the image");
        check(s.in_file_order && s.whole_arrays, "stream: pieces in file order, each array from its start to its end");
        check(s.largest_piece <= (16u << 20) && s.pieces > 1, "stream: no piece larger than one block");
        // the layout locates every array: the bytes at its offset are the array's
        const auto at = [&](const SessionFileArray& a) { return std::vector<uint8_t>(image.begin() + (std::ptrdiff_t) a.offset,
                                                                                    image.begin() + (std::ptrdiff_t) (a.offset + a.bytes)); };
        check(at(s.layout.live.state[0]) == original.live.gdn && at(s.layout.checkpoints[1].state[3]) == original.checkpoints[1].dead,
              "stream: the layout's offsets point at the arrays");
        std::vector<uint8_t> k0((size_t) original.kv[0].k.size());
        original.kv[0].k.read(k0.data(), 0, k0.size());
        check(at(s.layout.kv[0].parts[0]) == k0, "stream: a K part over 16 MiB located exactly");
        check(s.layout.live.ids == original.live.ids && s.layout.checkpoints.size() == 2 &&
              s.layout.checkpoints[0].ids == original.checkpoints[0].ids && s.layout.live.used == original.live.used,
              "stream: the token lists and stamps are read in pass 1");
        check(s.layout.kv[2].parts[4].bytes == 0 && s.layout.kv[1].parts[2].bytes == 0, "stream: empty parts are located too");
    }
    // unbuffered I/O (Windows FILE_FLAG_NO_BUFFERING where the volume allows it; POSIX O_DIRECT) reads the same
    {
        SessionStreamOptions opt;
        opt.unbuffered = true;
        Streamed s;
        check(stream_read(good, id, s, error, {}, opt) && same(s.image, original), "stream: an unbuffered read");
    }
    // within the exact runtime limits; and the limits refuse what get_payload's parse refuses
    {
        Streamed s;
        check(stream_read(good, id, s, error, exact(original)) && same(s.image, original), "limits: the exact limits read");
        auto l = exact(original); l.max_tokens = 999;
        check(!stream_read(good, id, s, error, l) && error.find("limit") != std::string::npos, "limits: a token count over the limit");
        l = exact(original); l.max_state_bytes[0] = 4098;
        check(!stream_read(good, id, s, error, l) && error.find("limit") != std::string::npos, "limits: a GDN state over the limit");
        l = exact(original); l.max_kv_bytes[0][0] = original.kv[0].k.size() - 1;
        check(!stream_read(good, id, s, error, l) && error.find("limit") != std::string::npos, "limits: a K part over the limit");
        l = exact(original); (*l.geometry)[3] ^= 1;
        check(!stream_read(good, id, s, error, l) && error.find("geometry") != std::string::npos, "limits: another geometry");
        l = exact(original); l.layer_range = std::make_pair(int64_t(0), int64_t(24));
        check(!stream_read(good, id, s, error, l) && error.find("layer") != std::string::npos, "limits: another layer range");
    }
    // the RAM preflight of pass 1: the block and the token lists, not the file
    {
        auto l = exact(original);
        uint64_t asked = 0;
        SessionStatus st;
        l.admit = [&](uint64_t n, std::string& why) { asked = n; why = "no RAM"; return false; };
        Streamed s;
        check(!stream_read(good, id, s, error, l, {}, &st) && st.error == SessionError::memory, "admit: a refusal is 'memory'");
        check(asked == session_stream_read_peak_bytes(image.size(), l) &&
              asked == (16u << 20) + (1u << 20) + 3u * 1000u * 4u, "admit: asked for the block, 1 MiB and three token lists");
        check(session_stream_read_peak_bytes(image.size(), SessionReadLimits{}) == (16u << 20) + (1u << 20) + image.size(),
              "admit: open limits ask for the file at most");
        bool admitted = false;
        l.admit = [&](uint64_t, std::string&) { admitted = true; return true; };
        check(!stream_read(good, {id.model ^ 1, id.config}, s, error, l) && !admitted, "admit: not asked for a foreign file");
        check(stream_read(good, id, s, error, l) && admitted && same(s.image, original), "admit: an admitted file reads");
    }
    // every refusal of pass 1 comes before the sink sees a byte
    {
        Streamed s;
        SessionStatus st;
        check(!stream_read(good, {id.model ^ 1, id.config}, s, error, {}, {}, &st) && error.find("model") != std::string::npos &&
              st.error == SessionError::invalid, "refuse: another model");
        check(!stream_read(good, {id.model, id.config ^ 1}, s, error) && error.find("config") != std::string::npos,
              "refuse: another configuration");
        for (size_t at : {size_t(64 + 300), image.size() / 2, size_t(20), image.size() - 12, image.size() - 3}) {
            auto d = image; d[at] ^= 0x40;
            const fs::path p = dir / "flip.bin"; spit(p, d);
            check(!stream_read(p, id, s, error), "refuse: a corrupted byte (the payload hash in pass 1)");
        }
        for (size_t cut : {size_t(0), size_t(63), size_t(200), image.size() / 2, image.size() - 1}) {
            const fs::path p = dir / "cut.bin";
            spit(p, std::vector<char>(image.begin(), image.begin() + (std::ptrdiff_t) cut));
            check(!stream_read(p, id, s, error), "refuse: a truncated file");
        }
        check(!stream_read(dir / "absent.bin", id, s, error), "refuse: a missing file");
    }
    // one delivery per verification; none without one; a failing sink stops it with its own reason
    {
        SessionFileStream file;
        SessionFileLayout layout;
        size_t n = 0;
        const SessionStreamSink count = [](const SessionStreamPiece&, const uint8_t*, size_t) { return true; };
        SessionStatus st;
        check(!file.deliver(count, error, &st) && st.error == SessionError::invalid, "deliver: nothing verified");
        check(file.verify(good.string(), id, {}, {}, layout, n, error) && file.deliver(count, error), "deliver: once");
        check(!file.deliver(count, error), "deliver: not twice");
        size_t calls = 0;
        const SessionStreamSink broken = [&](const SessionStreamPiece&, const uint8_t*, size_t) {
            if (++calls < 3) return true;
            error = "the device copy failed";
            return false;
        };
        check(file.verify(good.string(), id, {}, {}, layout, n, error), "deliver: verified again");
        check(!file.deliver(broken, error, &st) && error == "the device copy failed" && st.error == SessionError::io,
              "deliver: a failing sink stops it, its reason kept, 'io'");
        file.close();
        check(!file.deliver(count, error), "deliver: closed");
    }
    // progress over both passes: (done, 2 x size), never decreasing, a block apart at most, the first pass ending at
    // the file size and the second at twice it
    {
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        SessionStreamOptions opt;
        opt.progress = [&](uint64_t d, uint64_t t) { seen.push_back({d, t}); };
        Streamed s;
        check(stream_read(good, id, s, error, {}, opt), "progress: read");
        const uint64_t size = image.size();
        bool ok = !seen.empty() && seen.back().first == 2 * size;
        bool half = false;
        for (size_t i = 0; i < seen.size(); ++i) {
            ok = ok && seen[i].second == 2 * size && seen[i].first <= seen[i].second;
            if (i) ok = ok && seen[i].first >= seen[i - 1].first && seen[i].first - seen[i - 1].first <= (16u << 20);
            half = half || seen[i].first == size;
        }
        check(ok && half, "progress: monotonic over both passes, total twice the file, the first pass ends at its size");
    }
#ifndef _WIN32
    // a file changed between the passes (POSIX: the reader's handle does not keep writers out) fails at the end of pass 2
    {
        SessionFileStream file;
        SessionFileLayout layout;
        size_t n = 0;
        const fs::path p = dir / "changing.bin";
        spit(p, image);
        check(file.verify(p.string(), id, {}, {}, layout, n, error), "change: verified");
        auto d = image; d[image.size() / 2] ^= 0x01;
        spit(p, d);   // the same inode, rewritten in place
        const SessionStreamSink count = [](const SessionStreamPiece&, const uint8_t*, size_t) { return true; };
        SessionStatus st;
        check(!file.deliver(count, error, &st) && error.find("changed") != std::string::npos && st.error == SessionError::invalid,
              "change: pass 2 sees the changed payload");
    }
#endif

    // the bounded-RAM writer: the checkpoints from where they are, the GDN rows and the K/V from sources - the same
    // bytes as the ordinary writer, buffered and unbuffered
    {
        SavedConversation meta = original;
        meta.checkpoints.clear();
        meta.kv.clear();
        meta.live.gdn.clear();
        const std::vector<const ConversationCheckpoint*> cps = {&original.checkpoints[0], &original.checkpoints[1]};
        const SessionKvSource gdn = gdn_source(original.live.gdn);
        const std::vector<SessionKvSource> kv = kv_sources(original);
        for (bool unbuffered : {false, true}) {
            SessionWriteOptions opt;
            opt.unbuffered = unbuffered;
            SessionStatus st;
            const fs::path p = dir / (unbuffered ? "streamed-direct.bin" : "streamed.bin");
            check(session_file_write_streamed(p.string(), meta, cps, gdn, kv, id, written, error, opt, &st), "write: streamed");
            check(written == image.size() && slurp(p) == image, "write: byte-identical to the ordinary writer");
            check(no_temp(dir), "write: no temporary left");
            Streamed s;
            check(stream_read(p, id, s, error) && same(s.image, original), "write: the streamed file streams back");
            SavedConversation back;
            size_t n = 0;
            check(session_file_read(p.string(), id, back, n, error) && same(back, original), "write: the ordinary reader reads it");
        }
        // what it refuses: the parts in the image as well, a missing or layer-split checkpoint; nothing is left behind
        auto with_cp = meta; with_cp.checkpoints.push_back(original.checkpoints[0]);
        check(!session_file_write_streamed((dir / "x1.bin").string(), with_cp, cps, gdn, kv, id, written, error) &&
              !fs::exists(dir / "x1.bin"), "write: checkpoints in the image as well refused");
        auto with_gdn = meta; with_gdn.live.gdn = original.live.gdn;
        check(!session_file_write_streamed((dir / "x2.bin").string(), with_gdn, cps, gdn, kv, id, written, error),
              "write: GDN rows in the image as well refused");
        const std::vector<const ConversationCheckpoint*> null_cp = {nullptr};
        check(!session_file_write_streamed((dir / "x3.bin").string(), meta, null_cp, gdn, kv, id, written, error),
              "write: a missing checkpoint refused");
        ConversationCheckpoint split = original.checkpoints[0];
        split.stage_parts.push_back(checkpoint(1, 9));
        const std::vector<const ConversationCheckpoint*> split_cp = {&split};
        check(!session_file_write_streamed((dir / "x4.bin").string(), meta, split_cp, gdn, kv, id, written, error),
              "write: a layer-split checkpoint refused");
        // a GDN source that fails: no file, no temporary, the old file kept
        spit(dir / "keep.bin", std::vector<char>(5, 'k'));
        SessionKvSource broken = gdn;
        broken.read = [](size_t, size_t, void*, size_t) { return false; };
        check(!session_file_write_streamed((dir / "keep.bin").string(), meta, cps, broken, kv, id, written, error),
              "write: a failing GDN source fails the write");
        check(slurp(dir / "keep.bin") == std::vector<char>(5, 'k') && no_temp(dir), "write: the old file kept, no temporary");
        // no checkpoint at all: the same bytes as an image without them
        SavedConversation none = original;
        none.checkpoints.clear();
        check(session_file_write((dir / "none-a.bin").string(), none, id, written, error) &&
              session_file_write_streamed((dir / "none-b.bin").string(), meta, {}, gdn, kv, id, written, error) &&
              slurp(dir / "none-a.bin") == slurp(dir / "none-b.bin"), "write: without checkpoints, the same bytes");
    }
    // the RAM estimates: the streamed save copies no checkpoint and no GDN rows; pass 2 asks only for what the dropped
    // chain's storage does not hold
    {
        SessionSaveLive live;
        live.state_bytes = 5000; live.tokens = 1000; live.images = 1; live.kv_layers = 13;
        const uint64_t streamed = session_save_stream_peak_bytes(live);
        check(streamed >= (17u << 20) + 5000 + 4000 && streamed <= (17u << 20) + 5000 + 4000 + 16 + 14 * (sizeof(SessionKvSource) + 256),
              "estimate: the streamed save is the block, the small arrays, the tokens and the sources");
        ConversationCheckpoint fat = checkpoint(700, 3);
        fat.gdn.assign(8u << 20, 0x11);
        check(session_save_peak_bytes(&fat, live) > streamed + (16u << 20), "estimate: the ordinary save copies the checkpoint twice");
        SessionSaveLive huge = live;
        huge.state_bytes = UINT64_MAX;
        check(session_save_stream_peak_bytes(huge) == UINT64_MAX, "estimate: saturates");
        Streamed s;
        check(stream_read(good, id, s, error), "estimate: layout");
        const uint64_t small = 77 + 301 + 64 + 8, one = 4099 + small;
        check(session_stream_restore_peak_bytes(s.layout, 0) == (17u << 20) + small + 2 * one &&
              session_stream_restore_peak_bytes(s.layout, 1) == (17u << 20) + small + one &&
              session_stream_restore_peak_bytes(s.layout, 5) == (17u << 20) + small,
              "estimate: pass 2 asks for the checkpoints no dropped storage holds");
    }

    fs::remove_all(dir);
    std::printf("conversation_file_stream_test: %d checks passed\n", checks);
    return 0;
}
