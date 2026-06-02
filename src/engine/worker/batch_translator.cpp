#include "engine/worker/batch_translator.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include "common/error_code.h"
#include "engine/cache/kv_cache_manager.h"
#include "spdlog/spdlog.h"

namespace ccinfer {
namespace engine {

namespace {

bool check_seq_invariant(const SequenceState& seq, int block_size, int max_blocks) {
    if (block_size <= 0 || max_blocks <= 0) return false;
    if (seq.kv_written < 0 || seq.prompt_processed < 0) return false;
    if (seq.max_context_len <= 0) return false;
    if (seq.prompt_processed > seq.kv_written) return false;
    if (seq.kv_written > seq.max_context_len) return false;
    if (static_cast<std::size_t>(seq.prompt_processed) > seq.prompt_tokens.size()) return false;
    const std::size_t bt_sz = seq.block_table.size();
    const std::size_t b_sz = static_cast<std::size_t>(block_size);
    if (b_sz != 0 && bt_sz > std::numeric_limits<std::size_t>::max() / b_sz) return false;
    if (bt_sz * b_sz < static_cast<std::size_t>(seq.kv_written)) return false;
    int32_t sc = seq.block_table.shared_count();
    if (sc < 0 || sc > seq.block_table.size()) return false;
    if (static_cast<int64_t>(sc) * block_size > static_cast<int64_t>(seq.kv_written)) return false;
    if (static_cast<int64_t>(sc) * block_size > static_cast<int64_t>(seq.prompt_processed))
        return false;
    std::unordered_set<int32_t> seen;
    for (int b = 0; b < seq.block_table.size(); ++b) {
        int32_t bid = seq.block_table[b];
        if (bid < 0 || bid >= max_blocks) return false;
        if (!seen.insert(bid).second) return false;
    }
    return true;
}

}  // namespace

BatchTranslator::BatchTranslator(DefaultBackend& backend, KVCacheManager& kv_mgr, int block_size)
    : backend_(backend), kv_mgr_(kv_mgr), block_size_(block_size) {}

// ---------------------------------------------------------------------------
// translate
//
// Calling convention: if translate succeeds but the caller subsequently fails
// (forward or commit), the caller must invoke rollback(per_item) to release
// any KV blocks that were allocated.
// ---------------------------------------------------------------------------

Result<BatchTranslator::TranslateResult> BatchTranslator::translate(
    ScheduledBatch& batch, const std::unordered_map<SequenceId, SequenceState>& sequences) {
    if (block_size_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (batch.items.empty()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (batch.items.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    const int num_items = static_cast<int>(batch.items.size());
    std::vector<PerItemAlloc> per_item(static_cast<std::size_t>(num_items));

    auto fail = [&](ErrorCode ec) -> Result<TranslateResult> {
        rollback(per_item);
        return std::unexpected(ec);
    };

    // --- Phase 0: reject mixed Prefill/Decode before any KV allocation ---
    {
        bool has_prefill = false;
        bool has_decode = false;
        for (const auto& item : batch.items) {
            if (std::holds_alternative<PrefillChunk>(item))
                has_prefill = true;
            else
                has_decode = true;
        }
        if (has_prefill && has_decode) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
    }

    const int max_blk = kv_mgr_.max_blocks();

    // --- Phase 1: per-item validation and KV block allocation ---
    int total_tokens = 0;
    int batch_size = 0;
    int max_blocks_per_req = 0;
    std::unordered_set<SequenceId> seen_seq_ids;

    constexpr int kIntMax = std::numeric_limits<int>::max();

    for (int i = 0; i < num_items; ++i) {
        auto& item = batch.items[i];

        // Resolve seq_id before adjusting the item.
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);

        if (!seen_seq_ids.insert(seq_id).second) {
            return fail(ErrorCode::InvalidArgument);
        }

        auto it = sequences.find(seq_id);
        if (it == sequences.end()) {
            return fail(ErrorCode::InvalidArgument);
        }

        const auto& seq = it->second;
        if (seq.aborted || !check_seq_invariant(seq, block_size_, max_blk)) {
            return fail(ErrorCode::InvalidArgument);
        }

        // Adjust PrefillChunk for prefix-cache hits.  The scheduler does not
        // know which tokens were already cached; the worker adjusts the span
        // (or converts to a decode step if the entire prompt is cached).
        if (std::holds_alternative<PrefillChunk>(item)) {
            auto& pc = std::get<PrefillChunk>(item);
            if (pc.prompt_span.start < seq.prompt_processed) {
                int skip = seq.prompt_processed - pc.prompt_span.start;
                if (skip >= pc.prompt_span.length) {
                    // Entire chunk already cached.  Run a one-token bootstrap
                    // decode from the last prompt token to produce a sample,
                    // but do not commit that temporary KV slot; the next real
                    // decode will write the sampled token at the same logical
                    // position.
                    per_item[i].prefix_cache_bootstrap = true;
                    per_item[i].release_after_forward = true;
                    item = DecodeOneToken{seq_id, seq.prompt_tokens.back(),
                                          std::optional<int>(seq.kv_written)};
                } else {
                    pc.prompt_span.start += skip;
                    pc.prompt_span.length -= skip;
                }
            }
        }

        int new_tokens = 0;
        std::visit(
            [&](const auto& w) {
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    new_tokens = w.prompt_span.length;
                } else {
                    new_tokens = 1;
                }
            },
            item);

        per_item[i].kv_tokens_to_commit = new_tokens;
        per_item[i].prompt_tokens_to_commit =
            std::holds_alternative<PrefillChunk>(item) ? new_tokens : 0;
        if (per_item[i].prefix_cache_bootstrap) {
            per_item[i].kv_tokens_to_commit = 0;
            per_item[i].prompt_tokens_to_commit = 0;
        }

        if (std::holds_alternative<PrefillChunk>(item)) {
            const auto& pc = std::get<PrefillChunk>(item);
            if (pc.prompt_span.start < 0 || pc.prompt_span.length < 0) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (pc.prompt_span.start != seq.prompt_processed) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (pc.prompt_span.length > 0) {
                const std::size_t p_size = seq.prompt_tokens.size();
                if (static_cast<std::size_t>(pc.prompt_span.start) > p_size ||
                    static_cast<std::size_t>(pc.prompt_span.length) >
                        p_size - static_cast<std::size_t>(pc.prompt_span.start)) {
                    return fail(ErrorCode::InvalidArgument);
                }
                if (pc.expected_context_len.has_value() &&
                    *pc.expected_context_len != seq.kv_written) {
                    return fail(ErrorCode::InvalidArgument);
                }
                const std::size_t chunk_end = static_cast<std::size_t>(pc.prompt_span.start) +
                                              static_cast<std::size_t>(pc.prompt_span.length);
                for (std::size_t t = static_cast<std::size_t>(pc.prompt_span.start); t < chunk_end;
                     ++t) {
                    if (seq.prompt_tokens[t] < 0) return fail(ErrorCode::InvalidArgument);
                }
            }
        } else {
            const auto& d = std::get<DecodeOneToken>(item);
            if (d.input_token < 0) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (static_cast<std::size_t>(seq.prompt_processed) != seq.prompt_tokens.size()) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (d.expected_context_len.has_value() && *d.expected_context_len != seq.kv_written) {
                return fail(ErrorCode::InvalidArgument);
            }
        }

        const int64_t total_after = static_cast<int64_t>(seq.kv_written) + new_tokens;
        if (total_after > static_cast<int64_t>(seq.max_context_len)) {
            return fail(ErrorCode::RequestTooLong);
        }

        const int blocks_needed = static_cast<int>((total_after + block_size_ - 1) / block_size_);
        int blocks_owned = seq.block_table.size();

        BlockTable merged = seq.block_table;
        if (blocks_needed > blocks_owned) {
            int additional = blocks_needed - blocks_owned;
            auto alloc = kv_mgr_.allocate_blocks(additional);
            if (!alloc) {
                return fail(alloc.error());
            }

            per_item[i].new_blocks = *alloc;
            for (int b = 0; b < alloc->size(); ++b) {
                merged.push_back((*alloc)[b]);
            }
        }

        // Compute slot_mapping for the new tokens.
        per_item[i].slot_mapping.resize(static_cast<std::size_t>(new_tokens));
        for (int t = 0; t < new_tokens; ++t) {
            int global_pos = seq.kv_written + t;
            int block_idx = global_pos / block_size_;
            int pos_in_block = global_pos % block_size_;
            per_item[i].slot_mapping[static_cast<std::size_t>(t)] =
                merged[block_idx] * block_size_ + pos_in_block;
        }

        if (total_tokens > kIntMax - new_tokens) {
            return fail(ErrorCode::InvalidArgument);
        }
        total_tokens += new_tokens;
        max_blocks_per_req = std::max(max_blocks_per_req, merged.size());
        ++batch_size;
    }

    const std::size_t B_sz = static_cast<std::size_t>(batch_size);
    const std::size_t MBPR_sz = static_cast<std::size_t>(max_blocks_per_req);
    const std::size_t T_sz = static_cast<std::size_t>(total_tokens);
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

    // --- Phase 2: build host staging arrays ---
    if (MBPR_sz == 0) return fail(ErrorCode::InvalidArgument);
    // Overflow-check block_table_host size before allocation.
    if (B_sz > kMax / MBPR_sz) return fail(ErrorCode::InvalidArgument);

    std::vector<int32_t> token_ids_host(T_sz);
    std::vector<int32_t> positions_host(T_sz);
    std::vector<int32_t> slot_mapping_host(T_sz);
    std::vector<int32_t> block_table_host(B_sz * MBPR_sz, -1);
    std::vector<int32_t> query_start_loc(B_sz + 1);
    std::vector<int32_t> context_lens(B_sz);
    std::vector<int32_t> logits_indices(B_sz);

    int offset = 0;
    for (int i = 0; i < num_items; ++i) {
        const auto& item = batch.items[i];

        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        const auto& seq = sequences.at(seq_id);

        int new_tokens = static_cast<int>(per_item[i].slot_mapping.size());

        std::visit(
            [&](const auto& w) {
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    int start = w.prompt_span.start;
                    for (int t = 0; t < new_tokens; ++t) {
                        token_ids_host[static_cast<std::size_t>(offset + t)] =
                            seq.prompt_tokens[static_cast<std::size_t>(start + t)];
                    }
                } else {
                    token_ids_host[static_cast<std::size_t>(offset)] = w.input_token;
                }
            },
            item);

        for (int t = 0; t < new_tokens; ++t) {
            positions_host[static_cast<std::size_t>(offset + t)] = seq.kv_written + t;
            slot_mapping_host[static_cast<std::size_t>(offset + t)] =
                per_item[i].slot_mapping[static_cast<std::size_t>(t)];
        }

        BlockTable merged = seq.block_table;
        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            merged.push_back(per_item[i].new_blocks[b]);
        }
        for (int b = 0; b < merged.size(); ++b) {
            block_table_host[static_cast<std::size_t>(i) * MBPR_sz + static_cast<std::size_t>(b)] =
                merged[b];
        }

        query_start_loc[static_cast<std::size_t>(i)] = offset;
        context_lens[static_cast<std::size_t>(i)] = seq.kv_written + new_tokens;
        offset += new_tokens;
    }
    query_start_loc[B_sz] = offset;

    for (int i = 0; i < batch_size; ++i) {
        const int last_token_index = query_start_loc[static_cast<std::size_t>(i) + 1] - 1;

        if (auto* pf = std::get_if<PrefillChunk>(&batch.items[i])) {
            logits_indices[static_cast<std::size_t>(i)] = pf->needs_sample ? last_token_index : -1;
        } else {
            logits_indices[static_cast<std::size_t>(i)] = last_token_index;
        }
    }

    // --- Phase 3: allocate device buffers and upload ---
    TranslateResult result;

    auto& pb = result.physical_batch;
    // Recompute mode after Phase 1: prefix-cache hits may have converted
    // PrefillChunks to DecodeOneToken items.
    const ForwardMode mode = std::holds_alternative<PrefillChunk>(batch.items[0])
                                 ? ForwardMode::Prefill
                                 : ForwardMode::Decode;
    pb.num_tokens = total_tokens;
    pb.batch_size = batch_size;
    pb.max_blocks_per_req = max_blocks_per_req;
    pb.mode = mode;
    pb.item_indices.resize(B_sz);
    pb.item_seq_ids.resize(B_sz);
    pb.item_kinds.resize(B_sz);
    for (int i = 0; i < batch_size; ++i) {
        pb.item_indices[static_cast<std::size_t>(i)] = static_cast<std::size_t>(i);
        std::visit(
            [&](const auto& w) {
                pb.item_seq_ids[static_cast<std::size_t>(i)] = w.seq_id;
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    pb.item_kinds[static_cast<std::size_t>(i)] = WorkKind::PrefillChunk;
                } else {
                    pb.item_kinds[static_cast<std::size_t>(i)] = WorkKind::DecodeOneToken;
                }
            },
            batch.items[i]);
    }

    auto alloc_buf = [&](std::size_t bytes) -> Result<std::unique_ptr<DeviceBuffer>> {
        auto r = backend_.allocate_buffer(bytes);
        if (!r) return std::unexpected(r.error());
        return std::move(*r);
    };

    if (T_sz > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    {
        auto b = alloc_buf(T_sz * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.token_ids = std::move(*b);
    }
    {
        auto b = alloc_buf(T_sz * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.positions = std::move(*b);
    }
    {
        auto b = alloc_buf(T_sz * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.slot_mapping = std::move(*b);
    }
    if (B_sz * MBPR_sz > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    {
        auto b = alloc_buf(B_sz * MBPR_sz * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.block_table = std::move(*b);
    }
    if (B_sz + 1 > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    {
        auto b = alloc_buf((B_sz + 1) * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.query_start_loc = std::move(*b);
    }
    {
        auto b = alloc_buf(B_sz * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.context_lens = std::move(*b);
    }
    {
        auto b = alloc_buf(B_sz * sizeof(int32_t));
        if (!b) return fail(b.error());
        pb.logits_indices = std::move(*b);
    }

    auto r =
        backend_.memcpy_h2d(pb.token_ids->data(), token_ids_host.data(), T_sz * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }
    r = backend_.memcpy_h2d(pb.positions->data(), positions_host.data(), T_sz * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }
    r = backend_.memcpy_h2d(pb.slot_mapping->data(), slot_mapping_host.data(),
                            T_sz * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }
    r = backend_.memcpy_h2d(pb.block_table->data(), block_table_host.data(),
                            B_sz * MBPR_sz * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }
    r = backend_.memcpy_h2d(pb.query_start_loc->data(), query_start_loc.data(),
                            (B_sz + 1) * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }
    r = backend_.memcpy_h2d(pb.context_lens->data(), context_lens.data(), B_sz * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }
    r = backend_.memcpy_h2d(pb.logits_indices->data(), logits_indices.data(),
                            B_sz * sizeof(int32_t));
    if (!r) {
        backend_.synchronize();
        return fail(r.error());
    }

    auto sync_r = backend_.synchronize();
    if (!sync_r) return fail(sync_r.error());

    result.per_item = std::move(per_item);
    return result;
}

// ---------------------------------------------------------------------------
// commit
// ---------------------------------------------------------------------------

Result<void> BatchTranslator::commit(const ScheduledBatch& batch,
                                     std::unordered_map<SequenceId, SequenceState>& sequences,
                                     const std::vector<PerItemAlloc>& per_item) const {
    if (per_item.size() != batch.items.size()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const std::size_t num_items = batch.items.size();
    const int max_blk = kv_mgr_.max_blocks();

    // --- Phase 1: validate all items before mutating any SequenceState ---
    std::unordered_set<SequenceId> seen_seq_ids;
    std::vector<SequenceState*> to_update(num_items);
    std::vector<int> new_tokens_per_item(num_items);

    for (std::size_t i = 0; i < num_items; ++i) {
        const std::size_t slot_cnt = per_item[i].slot_mapping.size();
        if (slot_cnt > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return std::unexpected(ErrorCode::InvalidArgument);
        int new_tokens = static_cast<int>(slot_cnt);
        if (per_item[i].kv_tokens_to_commit < 0 || per_item[i].prompt_tokens_to_commit < 0 ||
            per_item[i].kv_tokens_to_commit > new_tokens ||
            per_item[i].prompt_tokens_to_commit > new_tokens) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        // Verify slot_mapping count matches WorkItem semantics.
        bool ok = false;
        std::visit(
            [&](const auto& w) {
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    ok = (new_tokens == w.prompt_span.length);
                } else {
                    ok = (new_tokens == 1);
                }
            },
            batch.items[i]);
        if (!ok || new_tokens <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, batch.items[i]);

        if (!seen_seq_ids.insert(seq_id).second) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        auto it = sequences.find(seq_id);
        if (it == sequences.end()) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        auto& seq = it->second;
        if (seq.aborted || !check_seq_invariant(seq, block_size_, max_blk)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        const int64_t total_after =
            static_cast<int64_t>(seq.kv_written) + per_item[i].kv_tokens_to_commit;
        if (total_after > static_cast<int64_t>(seq.max_context_len)) {
            return std::unexpected(ErrorCode::RequestTooLong);
        }

        if (std::holds_alternative<PrefillChunk>(batch.items[i])) {
            const auto& pc = std::get<PrefillChunk>(batch.items[i]);
            if (pc.prompt_span.start != seq.prompt_processed) {
                return std::unexpected(ErrorCode::InvalidArgument);
            }
            const std::size_t p_size = seq.prompt_tokens.size();
            if (static_cast<std::size_t>(pc.prompt_span.start) > p_size ||
                static_cast<std::size_t>(pc.prompt_span.length) >
                    p_size - static_cast<std::size_t>(pc.prompt_span.start)) {
                return std::unexpected(ErrorCode::InvalidArgument);
            }
            if (pc.expected_context_len.has_value() && *pc.expected_context_len != seq.kv_written) {
                return std::unexpected(ErrorCode::InvalidArgument);
            }
        } else {
            if (static_cast<std::size_t>(seq.prompt_processed) != seq.prompt_tokens.size()) {
                return std::unexpected(ErrorCode::InvalidArgument);
            }
            const auto& d = std::get<DecodeOneToken>(batch.items[i]);
            if (d.expected_context_len.has_value() && *d.expected_context_len != seq.kv_written) {
                return std::unexpected(ErrorCode::InvalidArgument);
            }
        }

        // Validate new_blocks: ids must be in range, non-duplicate, and not in
        // existing block_table.
        std::unordered_set<int32_t> existing;
        for (int b = 0; b < seq.block_table.size(); ++b) existing.insert(seq.block_table[b]);
        std::unordered_set<int32_t> new_seen;
        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            int32_t bid = per_item[i].new_blocks[b];
            if (bid < 0 || bid >= max_blk) return std::unexpected(ErrorCode::InvalidArgument);
            if (!new_seen.insert(bid).second) return std::unexpected(ErrorCode::InvalidArgument);
            if (existing.count(bid)) return std::unexpected(ErrorCode::InvalidArgument);
        }

        // Verify total blocks after commit cover kv_written + new_tokens.
        int64_t total_blocks_after =
            static_cast<int64_t>(seq.block_table.size()) + per_item[i].new_blocks.size();
        int64_t blocks_needed =
            (static_cast<int64_t>(seq.kv_written) + per_item[i].kv_tokens_to_commit +
             block_size_ - 1) /
            block_size_;
        if (total_blocks_after < blocks_needed) return std::unexpected(ErrorCode::InvalidArgument);

        to_update[i] = &seq;
        new_tokens_per_item[i] = new_tokens;
    }

    // --- Phase 2: mutate ---
    for (std::size_t i = 0; i < num_items; ++i) {
        auto& seq = *to_update[i];

        seq.kv_written += per_item[i].kv_tokens_to_commit;
        seq.prompt_processed += per_item[i].prompt_tokens_to_commit;

        if (!per_item[i].release_after_forward) {
            for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
                seq.block_table.push_back(per_item[i].new_blocks[b]);
            }
        } else if (per_item[i].new_blocks.size() > 0) {
            auto r = kv_mgr_.release_blocks(per_item[i].new_blocks);
            if (!r) return std::unexpected(r.error());
        }
    }

    return {};
}

// ---------------------------------------------------------------------------
// rollback
// ---------------------------------------------------------------------------

void BatchTranslator::rollback(const std::vector<PerItemAlloc>& per_item) const {
    for (const auto& alloc : per_item) {
        if (alloc.new_blocks.size() > 0) {
            auto r = kv_mgr_.release_blocks(alloc.new_blocks);
            if (!r) {
                spdlog::error("rollback: release_blocks failed for {} new blocks",
                              alloc.new_blocks.size());
            }
        }
    }
}

}  // namespace engine
}  // namespace ccinfer
