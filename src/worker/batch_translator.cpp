#include "worker/batch_translator.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include "base/error_code.h"
#include "cache/kv_cache_manager.h"
#include "spdlog/spdlog.h"

namespace ccinfer {

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

BatchTranslator::BatchTranslator(Backend& backend, KVCacheManager& kv_mgr, int block_size)
    : backend_(backend), kv_mgr_(kv_mgr), block_size_(block_size) {}

// Calling convention: if translate succeeds but the caller subsequently fails
// (forward or commit), the caller must invoke rollback(per_item) to release
// any KV blocks that were allocated.

Result<BatchTranslator::TranslateResult> BatchTranslator::translate(
    const ScheduledBatch& batch, const std::unordered_map<SequenceId, SequenceState>& sequences) {
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

    const int max_blk = kv_mgr_.max_blocks();

    // Phase 1: per-item validation and KV block allocation.
    int total_tokens = 0;
    int batch_size = 0;
    int max_blocks_per_req = 0;
    std::unordered_set<SequenceId> seen_seq_ids;

    constexpr int kIntMax = std::numeric_limits<int>::max();
    for (int i = 0; i < num_items; ++i) {
        const auto& item = batch.items[i];

        // Resolve seq_id before validating the item.
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

        if (std::holds_alternative<PrefillChunk>(item)) {
            per_item[i].kv_tokens_to_commit = new_tokens;
            per_item[i].prompt_tokens_to_commit = new_tokens;
        } else if (const auto* d = std::get_if<DecodeOneToken>(&item)) {
            if (d->write_kv) {
                per_item[i].kv_tokens_to_commit = 1;
                per_item[i].prompt_tokens_to_commit = 0;
            } else {
                per_item[i].kv_tokens_to_commit = 0;
                per_item[i].prompt_tokens_to_commit = 0;
            }
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
        } else if (const auto* d = std::get_if<DecodeOneToken>(&item)) {
            if (d->input_token < 0) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (static_cast<std::size_t>(seq.prompt_processed) != seq.prompt_tokens.size()) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (d->expected_context_len.has_value() && *d->expected_context_len != seq.kv_written) {
                return fail(ErrorCode::InvalidArgument);
            }
        }

        const bool is_bootstrap = std::holds_alternative<DecodeOneToken>(item) &&
                                       !std::get<DecodeOneToken>(item).write_kv;
        const int64_t total_after =
            static_cast<int64_t>(seq.kv_written) + (is_bootstrap ? 0 : new_tokens);
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

        if (!is_bootstrap) {
            per_item[i].slot_mapping.resize(static_cast<std::size_t>(new_tokens));
            for (int t = 0; t < new_tokens; ++t) {
                int global_pos = seq.kv_written + t;
                int block_idx = global_pos / block_size_;
                int pos_in_block = global_pos % block_size_;
                per_item[i].slot_mapping[static_cast<std::size_t>(t)] =
                    merged[block_idx] * block_size_ + pos_in_block;
            }
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

    // Phase 2: build host staging arrays.
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
    std::vector<int32_t> item_token_counts(B_sz);
    int max_position_id = 0;

    int offset = 0;
    for (int i = 0; i < num_items; ++i) {
        const auto& item = batch.items[i];

        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        const auto& seq = sequences.at(seq_id);

        const bool is_bootstrap = std::holds_alternative<DecodeOneToken>(item) &&
                                       !std::get<DecodeOneToken>(item).write_kv;
        const int new_tokens =
            is_bootstrap ? 1 : static_cast<int>(per_item[i].slot_mapping.size());
        item_token_counts[static_cast<std::size_t>(i)] = new_tokens;

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

        if (is_bootstrap) {
            const int pos = seq.kv_written - 1;
            positions_host[static_cast<std::size_t>(offset)] = pos;
            slot_mapping_host[static_cast<std::size_t>(offset)] = -1;  // unused
            max_position_id = std::max(max_position_id, pos);
        } else {
            for (int t = 0; t < new_tokens; ++t) {
                const int pos = seq.kv_written + t;
                positions_host[static_cast<std::size_t>(offset + t)] = pos;
                slot_mapping_host[static_cast<std::size_t>(offset + t)] =
                    per_item[i].slot_mapping[static_cast<std::size_t>(t)];
                max_position_id = std::max(max_position_id, pos);
            }
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
        context_lens[static_cast<std::size_t>(i)] =
            is_bootstrap ? seq.kv_written : seq.kv_written + new_tokens;
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

    // Phase 3: allocate device buffers and upload.
    TranslateResult result;

    auto& pb = result.physical_batch;
    pb.item_token_counts = std::move(item_token_counts);
    pb.sample_flags.resize(B_sz);
    for (std::size_t i = 0; i < B_sz; ++i) {
        pb.sample_flags[i] = logits_indices[i] >= 0;
    }
    pb.max_position_id = max_position_id;
    // DecodeOneToken with write_kv=false is decode-shaped but never writes KV.
    bool has_prefill = false;
    bool has_decode = false;
    for (const auto& item : batch.items) {
        if (std::holds_alternative<PrefillChunk>(item))
            has_prefill = true;
        else
            has_decode = true;
    }
    const ForwardMode mode = has_prefill && has_decode
                                 ? ForwardMode::Mixed
                                 : (has_prefill ? ForwardMode::Prefill : ForwardMode::Decode);
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

    if (T_sz > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    const std::int64_t T64 = static_cast<std::int64_t>(total_tokens);
    const std::int64_t B64 = static_cast<std::int64_t>(batch_size);
    const std::int64_t MBPR64 = static_cast<std::int64_t>(max_blocks_per_req);

    auto token_ids = Tensor::from_host(backend_, token_ids_host.data(), ops::DType::kInt32, {T64});
    if (!token_ids) return fail(token_ids.error());
    pb.token_ids = std::move(*token_ids);
    auto positions = Tensor::from_host(backend_, positions_host.data(), ops::DType::kInt32, {T64});
    if (!positions) return fail(positions.error());
    pb.positions = std::move(*positions);
    auto slot_mapping =
        Tensor::from_host(backend_, slot_mapping_host.data(), ops::DType::kInt32, {T64});
    if (!slot_mapping) return fail(slot_mapping.error());
    pb.slot_mapping = std::move(*slot_mapping);

    if (B_sz * MBPR_sz > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    auto block_table =
        Tensor::from_host(backend_, block_table_host.data(), ops::DType::kInt32, {B64, MBPR64});
    if (!block_table) return fail(block_table.error());
    pb.block_table = std::move(*block_table);

    if (B_sz + 1 > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    auto query_start_loc_dev =
        Tensor::from_host(backend_, query_start_loc.data(), ops::DType::kInt32, {B64 + 1});
    if (!query_start_loc_dev) return fail(query_start_loc_dev.error());
    pb.query_start_loc = std::move(*query_start_loc_dev);
    auto context_lens_dev =
        Tensor::from_host(backend_, context_lens.data(), ops::DType::kInt32, {B64});
    if (!context_lens_dev) return fail(context_lens_dev.error());
    pb.context_lens = std::move(*context_lens_dev);
    auto logits_indices_dev =
        Tensor::from_host(backend_, logits_indices.data(), ops::DType::kInt32, {B64});
    if (!logits_indices_dev) return fail(logits_indices_dev.error());
    pb.logits_indices = std::move(*logits_indices_dev);

    auto sync_r = backend_.synchronize();
    if (!sync_r) return fail(sync_r.error());

    result.per_item = std::move(per_item);
    return result;
}

Result<void> BatchTranslator::commit(const ScheduledBatch& batch,
                                     std::unordered_map<SequenceId, SequenceState>& sequences,
                                     const std::vector<PerItemAlloc>& per_item) const {
    if (per_item.size() != batch.items.size()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const std::size_t num_items = batch.items.size();
    const int max_blk = kv_mgr_.max_blocks();

    // Phase 1: validate all items before mutating any SequenceState.
    std::unordered_set<SequenceId> seen_seq_ids;
    std::vector<SequenceState*> to_update(num_items);

    for (std::size_t i = 0; i < num_items; ++i) {
        int new_tokens = 0;
        bool is_bootstrap = false;
        std::visit(
            [&](const auto& w) {
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    new_tokens = w.prompt_span.length;
                } else {
                    new_tokens = 1;
                    is_bootstrap = !w.write_kv;
                }
            },
            batch.items[i]);
        if (new_tokens <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        const std::size_t slot_cnt = per_item[i].slot_mapping.size();
        if (slot_cnt > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return std::unexpected(ErrorCode::InvalidArgument);
        if (!is_bootstrap && slot_cnt != static_cast<std::size_t>(new_tokens)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (per_item[i].kv_tokens_to_commit < 0 || per_item[i].prompt_tokens_to_commit < 0 ||
            per_item[i].kv_tokens_to_commit > new_tokens ||
            per_item[i].prompt_tokens_to_commit > new_tokens) {
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
        int64_t blocks_needed = (static_cast<int64_t>(seq.kv_written) +
                                 per_item[i].kv_tokens_to_commit + block_size_ - 1) /
                                block_size_;
        if (total_blocks_after < blocks_needed) return std::unexpected(ErrorCode::InvalidArgument);

        to_update[i] = &seq;
    }

    // Phase 2: mutate.
    for (std::size_t i = 0; i < num_items; ++i) {
        auto& seq = *to_update[i];

        seq.kv_written += per_item[i].kv_tokens_to_commit;
        seq.prompt_processed += per_item[i].prompt_tokens_to_commit;

        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            seq.block_table.push_back(per_item[i].new_blocks[b]);
        }
    }

    return {};
}

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

}  // namespace ccinfer
