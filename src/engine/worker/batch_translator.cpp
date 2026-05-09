#include "engine/worker/batch_translator.h"

#include <algorithm>
#include <type_traits>
#include <unordered_set>
#include <variant>

#include "common/error_code.h"
#include "engine/cache/kv_cache_manager.h"

namespace ccinfer {
namespace engine {

BatchTranslator::BatchTranslator(DefaultBackend& backend, KVCacheManager& kv_mgr, int block_size)
    : backend_(backend), kv_mgr_(kv_mgr), block_size_(block_size) {}

// ---------------------------------------------------------------------------
// translate
// ---------------------------------------------------------------------------

Result<BatchTranslator::TranslateResult> BatchTranslator::translate(
    const ScheduledBatch& batch,
    const std::unordered_map<SequenceId, SequenceState>& sequences) {

    if (block_size_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (batch.items.empty()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    int num_items = static_cast<int>(batch.items.size());
    std::vector<PerItemAlloc> per_item(num_items);

    // Unified failure path: rollback all prior allocations before returning.
    auto fail = [&](ErrorCode ec) -> Result<TranslateResult> {
        rollback(per_item);
        return std::unexpected(ec);
    };

    // --- Phase 1: per-item allocation and slot computation ---
    int total_tokens = 0;
    int batch_size = 0;
    int max_blocks_per_seq = 0;
    std::unordered_set<SequenceId> seen_seq_ids;

    for (int i = 0; i < num_items; ++i) {
        const auto& item = batch.items[i];

        SequenceId seq_id = 0;
        int new_tokens = 0;
        std::visit(
            [&](const auto& w) {
                seq_id = w.seq_id;
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    new_tokens = w.prompt_span.length;
                } else {
                    new_tokens = 1;
                }
            },
            item);

        if (!seen_seq_ids.insert(seq_id).second) {
            return fail(ErrorCode::InvalidArgument);
        }

        auto it = sequences.find(seq_id);
        if (it == sequences.end()) {
            return fail(ErrorCode::InvalidArgument);
        }

        const auto& seq = it->second;
        if (seq.aborted) {
            return fail(ErrorCode::InvalidArgument);
        }

        // Strict validation — translator does not silently fix scheduler errors.
        if (std::holds_alternative<PrefillChunk>(item)) {
            const auto& pc = std::get<PrefillChunk>(item);
            if (pc.prompt_span.length <= 0) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (pc.prompt_span.start != seq.prompt_processed) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (pc.prompt_span.start + pc.prompt_span.length >
                static_cast<int>(seq.prompt_tokens.size())) {
                return fail(ErrorCode::InvalidArgument);
            }
        } else {
            const auto& d = std::get<DecodeOneToken>(item);
            if (d.input_token < 0) {
                return fail(ErrorCode::InvalidArgument);
            }
        }

        int total_after = seq.kv_written + new_tokens;
        if (total_after > seq.max_context_len) {
            return fail(ErrorCode::RequestTooLong);
        }

        int blocks_needed = (total_after + block_size_ - 1) / block_size_;
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
        per_item[i].slot_mapping.resize(new_tokens);
        for (int t = 0; t < new_tokens; ++t) {
            int global_pos = seq.kv_written + t;
            int block_idx = global_pos / block_size_;
            int pos_in_block = global_pos % block_size_;
            per_item[i].slot_mapping[t] =
                merged[block_idx] * block_size_ + pos_in_block;
        }

        total_tokens += new_tokens;
        max_blocks_per_seq = std::max(max_blocks_per_seq, merged.size());
        ++batch_size;
    }

    // --- Phase 2: build host staging arrays ---
    std::vector<int32_t> token_ids_host(total_tokens);
    std::vector<int32_t> positions_host(total_tokens);
    std::vector<int32_t> slot_mapping_host(total_tokens);
    std::vector<int32_t> block_table_host(batch_size * max_blocks_per_seq, -1);
    std::vector<int32_t> query_start_loc(batch_size + 1);
    std::vector<int32_t> context_lens(batch_size);

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
                        token_ids_host[offset + t] = seq.prompt_tokens[start + t];
                    }
                } else {
                    token_ids_host[offset] = w.input_token;
                }
            },
            item);

        for (int t = 0; t < new_tokens; ++t) {
            positions_host[offset + t] = seq.kv_written + t;
            slot_mapping_host[offset + t] = per_item[i].slot_mapping[t];
        }

        BlockTable merged = seq.block_table;
        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            merged.push_back(per_item[i].new_blocks[b]);
        }
        for (int b = 0; b < merged.size(); ++b) {
            block_table_host[i * max_blocks_per_seq + b] = merged[b];
        }

        query_start_loc[i] = offset;
        context_lens[i] = seq.kv_written + new_tokens;
        offset += new_tokens;
    }
    query_start_loc[batch_size] = offset;

    // --- Phase 3: allocate device buffers and upload ---
    TranslateResult result;

    auto& pb = result.physical_batch;
    pb.num_tokens = total_tokens;
    pb.batch_size = batch_size;
    pb.max_blocks_per_seq = max_blocks_per_seq;
    pb.item_indices.resize(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        pb.item_indices[i] = static_cast<size_t>(i);
    }

    pb.token_ids = backend_.allocate_buffer(total_tokens * sizeof(int32_t));
    pb.positions = backend_.allocate_buffer(total_tokens * sizeof(int32_t));
    pb.slot_mapping = backend_.allocate_buffer(total_tokens * sizeof(int32_t));
    pb.block_table = backend_.allocate_buffer(static_cast<size_t>(batch_size * max_blocks_per_seq) *
                                              sizeof(int32_t));
    pb.query_start_loc =
        backend_.allocate_buffer(static_cast<size_t>(batch_size + 1) * sizeof(int32_t));
    pb.context_lens = backend_.allocate_buffer(batch_size * sizeof(int32_t));

    if (!pb.token_ids || !pb.positions || !pb.slot_mapping || !pb.block_table ||
        !pb.query_start_loc || !pb.context_lens) {
        return fail(ErrorCode::CudaOutOfMemory);
    }

    auto r = backend_.memcpy_h2d(pb.token_ids->data(), token_ids_host.data(),
                                  total_tokens * sizeof(int32_t));
    if (!r) return fail(r.error());
    r = backend_.memcpy_h2d(pb.positions->data(), positions_host.data(),
                             total_tokens * sizeof(int32_t));
    if (!r) return fail(r.error());
    r = backend_.memcpy_h2d(pb.slot_mapping->data(), slot_mapping_host.data(),
                             total_tokens * sizeof(int32_t));
    if (!r) return fail(r.error());
    r = backend_.memcpy_h2d(pb.block_table->data(), block_table_host.data(),
                             static_cast<size_t>(batch_size * max_blocks_per_seq) * sizeof(int32_t));
    if (!r) return fail(r.error());
    r = backend_.memcpy_h2d(pb.query_start_loc->data(), query_start_loc.data(),
                             static_cast<size_t>(batch_size + 1) * sizeof(int32_t));
    if (!r) return fail(r.error());
    r = backend_.memcpy_h2d(pb.context_lens->data(), context_lens.data(),
                             batch_size * sizeof(int32_t));
    if (!r) return fail(r.error());

    result.per_item = std::move(per_item);
    return result;
}

// ---------------------------------------------------------------------------
// commit
// ---------------------------------------------------------------------------

Result<void> BatchTranslator::commit(
    const ScheduledBatch& batch,
    std::unordered_map<SequenceId, SequenceState>& sequences,
    const std::vector<PerItemAlloc>& per_item) const {

    if (per_item.size() != batch.items.size()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    int num_items = static_cast<int>(batch.items.size());

    // --- Phase 1: validate all items before mutating any SequenceState ---
    std::unordered_set<SequenceId> seen_seq_ids;
    std::vector<SequenceState*> to_update(num_items);
    std::vector<int> new_tokens_per_item(num_items);

    for (int i = 0; i < num_items; ++i) {
        int new_tokens = static_cast<int>(per_item[i].slot_mapping.size());
        if (new_tokens <= 0) {
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
        if (seq.kv_written + new_tokens > seq.max_context_len) {
            return std::unexpected(ErrorCode::RequestTooLong);
        }

        to_update[i] = &seq;
        new_tokens_per_item[i] = new_tokens;
    }

    // --- Phase 2: mutate ---
    for (int i = 0; i < num_items; ++i) {
        auto& seq = *to_update[i];
        int new_tokens = new_tokens_per_item[i];

        seq.kv_written += new_tokens;

        if (std::holds_alternative<PrefillChunk>(batch.items[i])) {
            seq.prompt_processed += new_tokens;
        }

        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            seq.block_table.push_back(per_item[i].new_blocks[b]);
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
            if (!r) { /* TODO: log rollback release failure */
            }
        }
    }
}

}  // namespace engine
}  // namespace ccinfer
