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

int new_token_count(const WorkItem& item) {
    if (const auto* prefill = std::get_if<PrefillChunk>(&item)) return prefill->prompt_span.length;
    return 1;
}

bool is_bootstrap_decode(const WorkItem& item) {
    const auto* decode = std::get_if<DecodeOneToken>(&item);
    return decode != nullptr && !decode->write_kv;
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

void BatchTranslator::BatchExecutionPlan::commit() {
    assert(status_ == ExecutionPlanStatus::Prepared);
    assert(translator_ != nullptr && sequences_ != nullptr);

    translator_->commit_plan(batch_, *sequences_, per_item_);
    status_ = ExecutionPlanStatus::Committed;
}

void BatchTranslator::BatchExecutionPlan::rollback() noexcept {
    if (status_ != ExecutionPlanStatus::Prepared) return;
    if (translator_ != nullptr) translator_->rollback_allocations(per_item_);
    status_ = ExecutionPlanStatus::RolledBack;
}

Result<BatchTranslator::BatchExecutionPlan> BatchTranslator::prepare(
    const ScheduledBatch& batch, std::unordered_map<SequenceId, SequenceState>& sequences) {
    assert(block_size_ > 0);
    assert(!batch.items.empty());
    assert(batch.items.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()));

    auto plan = allocate_blocks(batch, sequences);
    if (!plan) return std::unexpected(plan.error());

    if (plan->batch_size == 0) {
        return make_plan(batch, &sequences, std::move(*plan), PhysicalBatch{});
    }

    auto physical_batch = build_physical_batch(batch, sequences, *plan);
    if (!physical_batch) return std::unexpected(physical_batch.error());

    return make_plan(batch, &sequences, std::move(*plan), std::move(*physical_batch));
}

Result<BatchTranslator::AllocationPlan> BatchTranslator::allocate_blocks(
    const ScheduledBatch& batch,
    const std::unordered_map<SequenceId, SequenceState>& sequences) const {
    const int num_items = static_cast<int>(batch.items.size());
    AllocationPlan plan;
    plan.per_item.resize(static_cast<std::size_t>(num_items));

    const int max_blk = kv_mgr_.max_blocks();
    constexpr int kIntMax = std::numeric_limits<int>::max();

    std::unordered_set<SequenceId> seen_seq_ids;
    seen_seq_ids.reserve(batch.items.size());

    for (int i = 0; i < num_items; ++i) {
        const auto& item = batch.items[i];
        PerItemAlloc& per_item = plan.per_item[static_cast<std::size_t>(i)];

        const SequenceId seq_id = work_sequence_id(item);
        assert(seen_seq_ids.insert(seq_id).second);

        auto it = sequences.find(seq_id);
        assert(it != sequences.end());
        const SequenceState& seq = it->second;
        assert(seq.kv_written >= 0 && seq.prompt_processed >= 0);
        assert(seq.max_context_len > 0);
        assert(seq.prompt_processed <= seq.kv_written);
        assert(seq.kv_written <= seq.max_context_len);
        assert(seq.prompt_processed <= seq.prompt_len);
        assert(static_cast<std::size_t>(seq.block_table.size()) *
                   static_cast<std::size_t>(block_size_) >=
               static_cast<std::size_t>(seq.kv_written));
        assert(seq.block_table.shared_count() >= 0 &&
               seq.block_table.shared_count() <= seq.block_table.size());
#ifndef NDEBUG
        {
            std::unordered_set<int32_t> seen;
            for (int b = 0; b < seq.block_table.size(); ++b) {
                const int32_t bid = seq.block_table[b];
                assert(bid >= 0 && bid < max_blk);
                assert(seen.insert(bid).second);
            }
        }
#endif

        const int new_tokens = new_token_count(item);
        const bool bootstrap = is_bootstrap_decode(item);

        // Logical commit increments.
        if (std::holds_alternative<PrefillChunk>(item)) {
            per_item.kv_tokens_to_commit = new_tokens;
            per_item.prompt_tokens_to_commit = new_tokens;
        } else if (const auto* decode = std::get_if<DecodeOneToken>(&item);
                   decode && decode->write_kv) {
            per_item.kv_tokens_to_commit = 1;
        }

        assert_item_matches_state(item, seq);

        const int64_t total_after =
            static_cast<int64_t>(seq.kv_written) + (bootstrap ? 0 : new_tokens);
        if (total_after > seq.max_context_len) {
            return std::unexpected(ErrorCode::RequestTooLong);
        }

        const int blocks_needed = static_cast<int>((total_after + block_size_ - 1) / block_size_);
        const int blocks_owned = seq.block_table.size();

        BlockTable merged = seq.block_table;
        if (blocks_needed > blocks_owned) {
            auto alloc = kv_mgr_.allocate_blocks(blocks_needed - blocks_owned);
            if (!alloc) {
                if (alloc.error() != ErrorCode::KVBlockExhausted) {
                    return std::unexpected(alloc.error());
                }
                if (std::holds_alternative<PrefillChunk>(item)) {
                    per_item.deferred = true;
                    continue;
                }
                const auto& decode = std::get<DecodeOneToken>(item);
                const int unwritten =
                    seq.tokens_generated - (seq.kv_written - seq.prompt_processed);
                if (decode.write_kv && unwritten < 2) {
                    per_item.force_no_write = true;
                } else {
                    per_item.deferred = true;
                    continue;
                }
            } else {
                per_item.new_blocks = *alloc;
                for (int b = 0; b < alloc->size(); ++b) merged.push_back((*alloc)[b]);
            }
        }

        const bool no_write = bootstrap || per_item.force_no_write;
        if (!no_write) {
            per_item.slot_mapping.resize(static_cast<std::size_t>(new_tokens));
            for (int t = 0; t < new_tokens; ++t) {
                const int global_pos = seq.kv_written + t;
                per_item.slot_mapping[static_cast<std::size_t>(t)] =
                    merged[global_pos / block_size_] * block_size_ + global_pos % block_size_;
            }
        }

        assert(new_tokens <= kIntMax - plan.total_tokens);
        plan.total_tokens += new_tokens;
        plan.max_blocks_per_req = std::max(plan.max_blocks_per_req, merged.size());
        ++plan.batch_size;
    }

    return plan;
}

void BatchTranslator::assert_item_matches_state(const WorkItem& item, const SequenceState& seq) {
    if (const auto* prefill = std::get_if<PrefillChunk>(&item)) {
        assert(prefill->prompt_span.start >= 0 && prefill->prompt_span.length >= 0);
        assert(prefill->prompt_span.start == seq.prompt_processed);
        assert(prefill->prompt_span.start <= seq.prompt_len);
        assert(prefill->prompt_span.length <= seq.prompt_len - prefill->prompt_span.start);
        assert(!prefill->expected_context_len.has_value() ||
               *prefill->expected_context_len == seq.kv_written);
        assert(prefill->tokens.size() == static_cast<std::size_t>(prefill->prompt_span.length));
        for (int32_t token : prefill->tokens) assert(token >= 0);
    } else {
        const auto& decode = std::get<DecodeOneToken>(item);
        assert(decode.input_token >= 0);
        assert(seq.prompt_processed == seq.prompt_len);
        assert(!decode.expected_context_len.has_value() ||
               *decode.expected_context_len == seq.kv_written);
    }
}

Result<PhysicalBatch> BatchTranslator::build_physical_batch(
    const ScheduledBatch& batch, const std::unordered_map<SequenceId, SequenceState>& sequences,
    const AllocationPlan& plan) const {
    const std::size_t B_sz = static_cast<std::size_t>(plan.batch_size);
    const std::size_t MBPR_sz = static_cast<std::size_t>(plan.max_blocks_per_req);
    const std::size_t T_sz = static_cast<std::size_t>(plan.total_tokens);
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

    assert(MBPR_sz > 0);
    assert(B_sz <= kMax / MBPR_sz);
    assert(T_sz <= kMax / sizeof(int32_t));
    assert(B_sz * MBPR_sz <= kMax / sizeof(int32_t));
    assert(B_sz + 1 <= kMax / sizeof(int32_t));

    std::vector<int32_t> token_ids_host(T_sz);
    std::vector<int32_t> positions_host(T_sz);
    std::vector<int32_t> slot_mapping_host(T_sz);
    std::vector<int32_t> block_table_host(B_sz * MBPR_sz, -1);
    std::vector<int32_t> query_start_loc(B_sz + 1);
    std::vector<int32_t> context_lens(B_sz);
    std::vector<int32_t> logits_indices(B_sz);
    std::vector<int32_t> item_token_counts(B_sz);
    std::vector<std::size_t> included_indices;
    included_indices.reserve(B_sz);
    int max_position_id = 0;

    int offset = 0;
    int phys = 0;
    const int num_items = static_cast<int>(batch.items.size());

    for (int i = 0; i < num_items; ++i) {
        const PerItemAlloc& per_item = plan.per_item[static_cast<std::size_t>(i)];
        if (per_item.deferred) continue;

        const auto& item = batch.items[static_cast<std::size_t>(i)];
        const SequenceState& seq = sequences.at(work_sequence_id(item));

        const bool no_write = is_bootstrap_decode(item) || per_item.force_no_write;
        const int new_tokens = no_write ? 1 : static_cast<int>(per_item.slot_mapping.size());
        item_token_counts[static_cast<std::size_t>(phys)] = new_tokens;

        std::visit(
            [&](const auto& work) {
                using T = std::decay_t<decltype(work)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    for (int t = 0; t < new_tokens; ++t) {
                        token_ids_host[static_cast<std::size_t>(offset + t)] =
                            work.tokens[static_cast<std::size_t>(t)];
                    }
                } else {
                    token_ids_host[static_cast<std::size_t>(offset)] = work.input_token;
                }
            },
            item);

        if (no_write) {
            const int pos = is_bootstrap_decode(item) ? seq.kv_written - 1 : seq.kv_written;
            positions_host[static_cast<std::size_t>(offset)] = pos;
            slot_mapping_host[static_cast<std::size_t>(offset)] = -1;
            max_position_id = std::max(max_position_id, pos);
        } else {
            for (int t = 0; t < new_tokens; ++t) {
                const int pos = seq.kv_written + t;
                positions_host[static_cast<std::size_t>(offset + t)] = pos;
                slot_mapping_host[static_cast<std::size_t>(offset + t)] =
                    per_item.slot_mapping[static_cast<std::size_t>(t)];
                max_position_id = std::max(max_position_id, pos);
            }
        }

        BlockTable merged = seq.block_table;
        for (int b = 0; b < per_item.new_blocks.size(); ++b) {
            merged.push_back(per_item.new_blocks[b]);
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
    for (int i = 0; i < plan.batch_size; ++i) {
        const std::size_t original_index = included_indices[static_cast<std::size_t>(i)];
        const int last_token_index = query_start_loc[static_cast<std::size_t>(i) + 1] - 1;
        const bool sample = std::holds_alternative<PrefillChunk>(batch.items[original_index])
                                ? std::get<PrefillChunk>(batch.items[original_index]).needs_sample
                                : true;
        if (sample) {
            logits_indices[static_cast<std::size_t>(i)] = num_logits++;
            logits_rows_host.push_back(last_token_index);
        } else {
            logits_indices[static_cast<std::size_t>(i)] = -1;
        }
    }

    PhysicalBatch pb;
    pb.item_token_counts = std::move(item_token_counts);
    pb.sample_flags.resize(B_sz);
    for (std::size_t i = 0; i < B_sz; ++i) {
        pb.sample_flags[i] = logits_indices[i] >= 0;
    }
    pb.logits_rows_host = std::move(logits_rows_host);
    pb.num_logits = num_logits;
    pb.max_position_id = max_position_id;

    bool has_prefill = false;
    bool has_decode = false;
    for (const std::size_t original_index : included_indices) {
        if (std::holds_alternative<PrefillChunk>(batch.items[original_index])) {
            has_prefill = true;
        } else {
            has_decode = true;
        }
    }
    pb.mode = has_prefill && has_decode ? ForwardMode::Mixed
              : has_prefill             ? ForwardMode::Prefill
                                        : ForwardMode::Decode;

    pb.num_tokens = plan.total_tokens;
    pb.batch_size = plan.batch_size;
    pb.max_blocks_per_req = plan.max_blocks_per_req;
    pb.item_indices.resize(B_sz);
    pb.item_seq_ids.resize(B_sz);
    pb.item_kinds.resize(B_sz);
    for (int i = 0; i < plan.batch_size; ++i) {
        const std::size_t original_index = included_indices[static_cast<std::size_t>(i)];
        pb.item_indices[static_cast<std::size_t>(i)] = original_index;
        pb.item_seq_ids[static_cast<std::size_t>(i)] =
            work_sequence_id(batch.items[original_index]);
        pb.item_kinds[static_cast<std::size_t>(i)] = work_kind(batch.items[original_index]);
    }

    const std::int64_t T64 = static_cast<std::int64_t>(plan.total_tokens);
    const std::int64_t B64 = static_cast<std::int64_t>(plan.batch_size);
    const std::int64_t MBPR64 = static_cast<std::int64_t>(plan.max_blocks_per_req);

    auto token_ids = Tensor::from_host(backend_, token_ids_host.data(), ccop::DType::kInt32, {T64});
    if (!token_ids) return std::unexpected(token_ids.error());
    pb.token_ids = std::move(*token_ids);

    auto positions = Tensor::from_host(backend_, positions_host.data(), ccop::DType::kInt32, {T64});
    if (!positions) return std::unexpected(positions.error());
    pb.positions = std::move(*positions);

    auto slot_mapping =
        Tensor::from_host(backend_, slot_mapping_host.data(), ccop::DType::kInt32, {T64});
    if (!slot_mapping) return std::unexpected(slot_mapping.error());
    pb.slot_mapping = std::move(*slot_mapping);

    auto block_table =
        Tensor::from_host(backend_, block_table_host.data(), ccop::DType::kInt32, {B64, MBPR64});
    if (!block_table) return std::unexpected(block_table.error());
    pb.block_table = std::move(*block_table);

    auto query_start_loc_dev =
        Tensor::from_host(backend_, query_start_loc.data(), ccop::DType::kInt32, {B64 + 1});
    if (!query_start_loc_dev) return std::unexpected(query_start_loc_dev.error());
    pb.query_start_loc = std::move(*query_start_loc_dev);

    auto context_lens_dev =
        Tensor::from_host(backend_, context_lens.data(), ccop::DType::kInt32, {B64});
    if (!context_lens_dev) return std::unexpected(context_lens_dev.error());
    pb.context_lens = std::move(*context_lens_dev);

    auto logits_indices_dev =
        Tensor::from_host(backend_, logits_indices.data(), ccop::DType::kInt32, {B64});
    if (!logits_indices_dev) return std::unexpected(logits_indices_dev.error());
    pb.logits_indices = std::move(*logits_indices_dev);

    auto sync_r = backend_.synchronize();
    if (!sync_r) return std::unexpected(sync_r.error());

    return pb;
}

BatchTranslator::BatchExecutionPlan BatchTranslator::make_plan(
    ScheduledBatch batch, std::unordered_map<SequenceId, SequenceState>* sequences,
    AllocationPlan&& plan, PhysicalBatch physical_batch) {
    BatchExecutionPlan exec_plan(this, std::move(batch), sequences, std::move(physical_batch),
                                 std::move(plan.per_item));

    for (std::size_t i = 0; i < exec_plan.per_item_.size(); ++i) {
        if (exec_plan.per_item_[i].deferred) exec_plan.deferred_indices.push_back(i);
        exec_plan.no_write_flags.push_back(exec_plan.per_item_[i].force_no_write);
    }
    return exec_plan;
}

void BatchTranslator::commit_plan(const ScheduledBatch& batch,
                                  std::unordered_map<SequenceId, SequenceState>& sequences,
                                  const std::vector<PerItemAlloc>& per_item) const {
    assert(per_item.size() == batch.items.size());

    for (std::size_t i = 0; i < batch.items.size(); ++i) {
        const SequenceId seq_id = work_sequence_id(batch.items[i]);
        auto it = sequences.find(seq_id);
        assert(it != sequences.end() && "commit_plan called with unknown sequence");
        SequenceState& seq = it->second;

        seq.kv_written += per_item[i].kv_tokens_to_commit;
        seq.prompt_processed += per_item[i].prompt_tokens_to_commit;
        for (int b = 0; b < per_item[i].new_blocks.size(); ++b) {
            seq.block_table.push_back(per_item[i].new_blocks[b]);
        }
    }
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
