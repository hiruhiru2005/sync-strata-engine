// Host-only opt-in grouping permutation. No HIP calls, device math, or precision changes.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill::detail {

inline bool mmq_resident_sort_flag(const char* value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

inline bool mmq_resident_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_RESIDENT_SORT_NE"));
    return on;
}

// STRATA_MMQ_GROUP_SORT=1: the same row-count order for the full 16-expert groups only; the last, partial group keeps
// its experts in ID order. A full group's launch has 160 * ntx (gate/up) or 320 * ntx (down) tiles, at or above the
// 90 % tile efficiency on the cards' SM counts, so it runs one block per tile with no stream-k K split whatever its tile
// width: each expert's rows get the same bits in any full group. The partial group can take the stream-k split, whose
// K boundaries depend on the group's tile count - leaving it unchanged keeps the prompt's bits.
inline bool mmq_group_sort_requested() {
    static const bool on = mmq_resident_sort_flag(std::getenv("STRATA_MMQ_GROUP_SORT"));
    return on;
}

// How many of `n` ordered experts the sort may move: all of them (STRATA_MMQ_RESIDENT_SORT_NE), or the full groups
// of `group` experts (STRATA_MMQ_GROUP_SORT).
inline size_t mmq_resident_sort_span(size_t n, size_t group, bool keep_tail) {
    return keep_tail && group > 0 ? n - n % group : n;
}

inline bool mmq_resident_sort_eligible(bool requested, bool use_mmq, bool native_layout, bool has_cache,
                                      const int32_t* layer_resident, int64_t n_expert,
                                      bool stream_all, bool layer_stream_empty) {
    // native-layout packs too: slot/src/off are built the same way for both layouts, and nothing reads off[e + 1] as a
    // count (4x GP100, sm_60: 1.2K prompt 7.27 -> 5.32 s, 35K 63.3 -> 54.1 s, the same bits)
    (void) native_layout;
    if (!requested || !use_mmq || !has_cache || !layer_resident || n_expert <= 0) return false;
    // All experts, including unselected ones: leave every partial/streamed layer in ID order.
    if (stream_all && !layer_stream_empty) return false;
    for (int64_t e = 0; e < n_expert; ++e) if (layer_resident[e] < 0) return false;
    return true;
}

// Preconditions are established by the existing caller: IDs were range-checked and counted;
// order contains each positive-count expert once in ID order; off has count.size()+1 entries;
// routes fits its original int32 offsets, and slot/src each have routes writable entries.
// After sorting, off[e] is an expert-indexed START, not an ID-prefix end at off[e+1].
// Only order[0, span) is sorted (span >= order.size(): all of it); the entries after it keep their positions.
inline void mmq_resident_sort_rows(const int32_t* ids, int64_t routes, int32_t k,
                                   const std::vector<int32_t>& count, std::vector<int32_t>& off,
                                   std::vector<int32_t>& order, int32_t* slot, int32_t* src,
                                   size_t span = SIZE_MAX) {
    std::stable_sort(order.begin(), order.begin() + (ptrdiff_t) std::min(span, order.size()), [&](int32_t a, int32_t b) {
        return count[(size_t) a] != count[(size_t) b]
            ? count[(size_t) a] < count[(size_t) b] : a < b;
    });
    std::fill(off.begin(), off.end(), 0);
    int32_t cursor = 0;
    for (int32_t e : order) {
        off[(size_t) e] = cursor;
        cursor += count[(size_t) e];
    }
    off.back() = cursor;
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    // Preserve token-major/rank-major traversal, hence the original within-expert row order.
    for (int64_t i = 0; i < routes; ++i) {
        const int32_t e = ids[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / k);
    }
}

// QuantLab (STRATA_QL_MMQ_MIXED=1): a per-expert K (mixed) layer's MMQ groups.  MMQ takes one type per launch, so the
// routed experts are taken in (class, id) order - `corder`, the order their rows are laid out in - and cut into groups
// of one class and at most `group` experts.  The walk still meets the experts in id order (its stream plan is in id
// order): each one is gathered into its group's slot, and a group launches once all of its experts are in.  Within a
// class the experts come in id order too, so a class has at most one group filling at a time (its slot region).
struct MixedGroups {
    std::vector<int32_t> corder;                    // the routed experts by (class, id)
    std::vector<int32_t> pos;                       // expert -> its position in corder (-1: not routed in this chunk)
    std::vector<int32_t> group_of;                  // corder position -> its group
    std::vector<int32_t> gstart, gsize, gclass;     // group g: corder[gstart, gstart + gsize), all of class gclass
    std::vector<int32_t> gfill;                     // the experts gathered into group g so far
};
// Preconditions as mmq_resident_sort_rows': order holds each positive-count expert once (every routed expert: no CPU
// or peer share), off has count.size() + 1 entries, slot / src hold `routes` entries.  cls[e] is expert e's class.
inline void mmq_class_rows(const int32_t* ids, int64_t routes, int32_t k, const std::vector<int32_t>& count,
                           std::vector<int32_t>& off, const std::vector<int32_t>& order, const uint8_t* cls, int32_t* slot,
                           int32_t* src, int group, MixedGroups& mg) {
    mg.corder = order;
    std::stable_sort(mg.corder.begin(), mg.corder.end(), [&](int32_t a, int32_t b) {
        return cls[a] != cls[b] ? cls[a] < cls[b] : a < b;
    });
    std::fill(off.begin(), off.end(), 0);
    int32_t cursor = 0;
    for (int32_t e : mg.corder) {
        off[(size_t) e] = cursor;
        cursor += count[(size_t) e];
    }
    off.back() = cursor;
    std::vector<int32_t> fill(off.begin(), off.end() - 1);
    for (int64_t i = 0; i < routes; ++i) {   // token-major, rank-major: the within-expert row order is unchanged
        const int32_t e = ids[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot[(size_t) i] = p;
        src[(size_t) p] = (int32_t) (i / k);
    }
    mg.pos.assign(count.size(), -1);
    mg.group_of.clear();
    mg.gstart.clear();
    mg.gsize.clear();
    mg.gclass.clear();
    for (size_t p = 0; p < mg.corder.size(); ++p) {
        const int32_t e = mg.corder[p];
        mg.pos[(size_t) e] = (int32_t) p;
        if (mg.gstart.empty() || mg.gclass.back() != (int32_t) cls[e] || mg.gsize.back() == group) {
            mg.gstart.push_back((int32_t) p);
            mg.gsize.push_back(0);
            mg.gclass.push_back((int32_t) cls[e]);
        }
        ++mg.gsize.back();
        mg.group_of.push_back((int32_t) mg.gstart.size() - 1);
    }
    mg.gfill.assign(mg.gstart.size(), 0);
}

}  // namespace strata::prefill::detail
