#include "worker/batch_translator.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

#include "cache/kv_cache_manager.h"
#include "common/error_code.h"
#include "facade/log.h"

namespace ccinfer {

namespace {

bool check_seq_invariant(const SequenceState& seq, int block_size, int max_blocks) {
    if (block_size <= 0 || max_blocks <= 0) return false;
    if (seq.kv_written < 0 || seq.prompt_processed < 0) return false;
    if (seq.max_context_len <= 0) return false;
    if (seq.prompt_processed > seq.kv_written) return false;
    if (seq.kv_written > seq.max_context_len) return false;
    if (seq.prompt_processed > seq.prompt_len) return false;
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

BatchTranslator::BatchExecutionPlan::BatchExecutionPlan(
    BatchTranslator* translator, ScheduledBatch batch,
    std::unordered_map<SequenceId, SequenceState>* sequences, PhysicalBatch physical_batch,
    std::vector<PerItemAlloc> per_item)
    : physical_batch(std::move(physical_batch)),
      translator_(translator),
      batch_(std::move(batch)),
      sequences_(sequences),
      per_item_(std::move(per_item)) {}

BatchTranslator::BatchExecutionPlan::BatchExecutionPlan(BatchExecutionPlan&& other) noexcept
    : physical_batch(std::move(other.physical_batch)),
      translator_(std::exchange(other.translator_, nullptr)),
      batch_(std::move(other.batch_)),
      sequences_(std::exchange(other.sequences_, nullptr)),
      per_item_(std::move(other.per_item_)),
      status_(std::exchange(other.status_, ExecutionPlanStatus::RolledBack)) {}

BatchTranslator::BatchExecutionPlan& BatchTranslator::BatchExecutionPlan::operator=(
    BatchExecutionPlan&& other) noexcept {
    if (this == &other) return *this;

    rollback();
    physical_batch = std::move(other.physical_batch);
    translator_ = std::exchange(other.translator_, nullptr);
    batch_ = std::move(other.batch_);
    sequences_ = std::exchange(other.sequences_, nullptr);
    per_item_ = std::move(other.per_item_);
    status_ = std::exchange(other.status_, ExecutionPlanStatus::RolledBack);
    return *this;
}

BatchTranslator::BatchExecutionPlan::~BatchExecutionPlan() { rollback(); }

Result<void> BatchTranslator::BatchExecutionPlan::commit() {
    if (status_ != ExecutionPlanStatus::Prepared) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (translator_ == nullptr || sequences_ == nullptr) {
        return std::unexpected(ErrorCode::InternalError);
    }

    auto result = translator_->commit_plan(batch_, *sequences_, per_item_);
    if (result) status_ = ExecutionPlanStatus::Committed;
    return result;
}

void BatchTranslator::BatchExecutionPlan::rollback() noexcept {
    if (status_ != ExecutionPlanStatus::Prepared) return;
    if (translator_ != nullptr) translator_->rollback_allocations(per_item_);
    status_ = ExecutionPlanStatus::RolledBack;
}

Result<BatchTranslator::BatchExecutionPlan> BatchTranslator::prepare(
    const ScheduledBatch& batch, std::unordered_map<SequenceId, SequenceState>& sequences) {
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

    auto fail = [&](ErrorCode ec) -> Result<BatchExecutionPlan> {
        rollback_allocations(per_item);
        return std::unexpected(ec);
    };
    auto mark_deferred = [&](int i) {
        per_item[i].deferred = true;
        per_item[i].kv_tokens_to_commit = 0;
        per_item[i].prompt_tokens_to_commit = 0;
    };
    auto mark_no_write = [&](int i) {
        per_item[i].force_no_write = true;
        per_item[i].kv_tokens_to_commit = 0;
        per_item[i].prompt_tokens_to_commit = 0;
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
        if (!check_seq_invariant(seq, block_size_, max_blk)) {
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
                if (pc.prompt_span.start < 0 || pc.prompt_span.start > seq.prompt_len ||
                    pc.prompt_span.length > seq.prompt_len - pc.prompt_span.start) {
                    return fail(ErrorCode::InvalidArgument);
                }
                if (pc.expected_context_len.has_value() &&
                    *pc.expected_context_len != seq.kv_written) {
                    return fail(ErrorCode::InvalidArgument);
                }
                if (pc.tokens.size() != static_cast<std::size_t>(pc.prompt_span.length)) {
                    return fail(ErrorCode::InvalidArgument);
                }
                for (int32_t token : pc.tokens) {
                    if (token < 0) return fail(ErrorCode::InvalidArgument);
                }
            }
        } else if (const auto* d = std::get_if<DecodeOneToken>(&item)) {
            if (d->input_token < 0) {
                return fail(ErrorCode::InvalidArgument);
            }
            if (seq.prompt_processed != seq.prompt_len) {
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
                if (alloc.error() == ErrorCode::KVBlockExhausted) {
                    if (std::holds_alternative<PrefillChunk>(item)) {
                        mark_deferred(i);
                        continue;
                    }
                    const auto& d = std::get<DecodeOneToken>(item);
                    const int unwritten =
                        seq.tokens_generated - (seq.kv_written - seq.prompt_processed);
                    if (d.write_kv && unwritten < 2) {
                        mark_no_write(i);
                    } else {
                        mark_deferred(i);
                        continue;
                    }
                } else {
                    return fail(alloc.error());
                }
            } else {
                per_item[i].new_blocks = *alloc;
                for (int b = 0; b < alloc->size(); ++b) {
                    merged.push_back((*alloc)[b]);
                }
            }
        }

        const bool no_write = is_bootstrap || per_item[i].force_no_write;
        if (!no_write) {
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
    int phys = 0;
    std::vector<std::size_t> included_indices;
    for (int i = 0; i < num_items; ++i) {
        if (per_item[i].deferred) continue;

        const auto& item = batch.items[i];

        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        const auto& seq = sequences.at(seq_id);

        const bool is_bootstrap = std::holds_alternative<DecodeOneToken>(item) &&
                                  !std::get<DecodeOneToken>(item).write_kv;
        const bool force_no_write = per_item[i].force_no_write;
        const bool no_write = is_bootstrap || force_no_write;
        const int new_tokens = no_write ? 1 : static_cast<int>(per_item[i].slot_mapping.size());
        item_token_counts[static_cast<std::size_t>(phys)] = new_tokens;

        std::visit(
            [&](const auto& w) {
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    for (int t = 0; t < new_tokens; ++t) {
                        token_ids_host[static_cast<std::size_t>(offset + t)] = w.tokens[static_cast<std::size_t>(t)];
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
        } else if (force_no_write) {
            const int pos = seq.kv_written;
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
            block_table_host[static_cast<std::size_t>(phys) * MBPR_sz +
                             static_cast<std::size_t>(b)] = merged[b];
        }

        query_start_loc[static_cast<std::size_t>(phys)] = offset;
        context_lens[static_cast<std::size_t>(phys)] =
            no_write ? seq.kv_written : seq.kv_written + new_tokens;
        included_indices.push_back(static_cast<std::size_t>(i));
        offset += new_tokens;
        ++phys;
    }
    query_start_loc[B_sz] = offset;

    std::vector<int32_t> logits_rows_host;
    int num_logits = 0;
    for (int i = 0; i < batch_size; ++i) {
        const std::size_t original_index = included_indices[static_cast<std::size_t>(i)];
        const int last_token_index = query_start_loc[static_cast<std::size_t>(i) + 1] - 1;
        const bool sample =
            std::holds_alternative<PrefillChunk>(batch.items[original_index])
                ? std::get<PrefillChunk>(batch.items[original_index]).needs_sample
                : true;
        if (sample) {
            logits_indices[static_cast<std::size_t>(i)] = num_logits++;
            logits_rows_host.push_back(last_token_index);
        } else {
            logits_indices[static_cast<std::size_t>(i)] = -1;
        }
    }

    // Phase 3: allocate device buffers and upload.
    PhysicalBatch physical_batch;

    auto& pb = physical_batch;
    pb.item_token_counts = std::move(item_token_counts);
    pb.sample_flags.resize(B_sz);
    for (std::size_t i = 0; i < B_sz; ++i) {
        pb.sample_flags[i] = logits_indices[i] >= 0;
    }
    pb.logits_rows_host = std::move(logits_rows_host);
    pb.num_logits = num_logits;
    pb.max_position_id = max_position_id;
    // DecodeOneToken with write_kv=false is decode-shaped but never writes KV.
    bool has_prefill = false;
    bool has_decode = false;
    for (const std::size_t original_index : included_indices) {
        const auto& item = batch.items[original_index];
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
        const std::size_t original_index = included_indices[static_cast<std::size_t>(i)];
        pb.item_indices[static_cast<std::size_t>(i)] = original_index;
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
            batch.items[original_index]);
    }

    std::vector<std::size_t> deferred_indices;
    std::vector<bool> no_write_flags(static_cast<std::size_t>(num_items), false);
    for (int i = 0; i < num_items; ++i) {
        if (per_item[i].deferred) deferred_indices.push_back(static_cast<std::size_t>(i));
        if (per_item[i].force_no_write) no_write_flags[static_cast<std::size_t>(i)] = true;
    }

    if (batch_size == 0) {
        BatchExecutionPlan plan(this, batch, &sequences, std::move(physical_batch),
                                std::move(per_item));
        plan.deferred_indices = std::move(deferred_indices);
        plan.no_write_flags = std::move(no_write_flags);
        return plan;
    }

    if (T_sz > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    const std::int64_t T64 = static_cast<std::int64_t>(total_tokens);
    const std::int64_t B64 = static_cast<std::int64_t>(batch_size);
    const std::int64_t MBPR64 = static_cast<std::int64_t>(max_blocks_per_req);

    auto token_ids = Tensor::from_host(backend_, token_ids_host.data(), ccop::DType::kInt32, {T64});
    if (!token_ids) return fail(token_ids.error());
    pb.token_ids = std::move(*token_ids);
    auto positions = Tensor::from_host(backend_, positions_host.data(), ccop::DType::kInt32, {T64});
    if (!positions) return fail(positions.error());
    pb.positions = std::move(*positions);
    auto slot_mapping =
        Tensor::from_host(backend_, slot_mapping_host.data(), ccop::DType::kInt32, {T64});
    if (!slot_mapping) return fail(slot_mapping.error());
    pb.slot_mapping = std::move(*slot_mapping);

    if (B_sz * MBPR_sz > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    auto block_table =
        Tensor::from_host(backend_, block_table_host.data(), ccop::DType::kInt32, {B64, MBPR64});
    if (!block_table) return fail(block_table.error());
    pb.block_table = std::move(*block_table);

    if (B_sz + 1 > kMax / sizeof(int32_t)) return fail(ErrorCode::InvalidArgument);
    auto query_start_loc_dev =
        Tensor::from_host(backend_, query_start_loc.data(), ccop::DType::kInt32, {B64 + 1});
    if (!query_start_loc_dev) return fail(query_start_loc_dev.error());
    pb.query_start_loc = std::move(*query_start_loc_dev);
    auto context_lens_dev =
        Tensor::from_host(backend_, context_lens.data(), ccop::DType::kInt32, {B64});
    if (!context_lens_dev) return fail(context_lens_dev.error());
    pb.context_lens = std::move(*context_lens_dev);
    auto logits_indices_dev =
        Tensor::from_host(backend_, logits_indices.data(), ccop::DType::kInt32, {B64});
    if (!logits_indices_dev) return fail(logits_indices_dev.error());
    pb.logits_indices = std::move(*logits_indices_dev);

    auto sync_r = backend_.synchronize();
    if (!sync_r) return fail(sync_r.error());

    BatchExecutionPlan plan(this, batch, &sequences, std::move(physical_batch),
                            std::move(per_item));
    plan.deferred_indices = std::move(deferred_indices);
    plan.no_write_flags = std::move(no_write_flags);
    return plan;
}

Result<void> BatchTranslator::commit_plan(const ScheduledBatch& batch,
                                          std::unordered_map<SequenceId, SequenceState>& sequences,
                                          const std::vector<PerItemAlloc>& per_item) const {
    if (per_item.size() != batch.items.size()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    // prepare() already validated every item against the same private
    // SequenceState copy. commit() only applies the prepared increments.
    for (std::size_t i = 0; i < batch.items.size(); ++i) {
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, batch.items[i]);

        auto it = sequences.find(seq_id);
        assert(it != sequences.end() && "commit_plan called with unknown sequence");
        auto& seq = it->second;

        seq.kv_written += per_item[i].kv_tokens_to_commit;
        seq.prompt_processed += per_item[i].prompt_tokens_to_commit;
        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            seq.block_table.push_back(per_item[i].new_blocks[b]);
        }
    }

    return {};
}

void BatchTranslator::rollback_allocations(
    const std::vector<PerItemAlloc>& per_item) const noexcept {
    for (const auto& alloc : per_item) {
        if (alloc.new_blocks.size() > 0) {
            auto r = kv_mgr_.release_blocks(alloc.new_blocks);
            if (!r) {
                ccLog::error("rollback: release_blocks failed for {} new blocks",
                             alloc.new_blocks.size());
            }
        }
    }
}

}  // namespace ccinfer
