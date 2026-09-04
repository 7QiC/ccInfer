#include "worker/worker.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>

#include <boost/asio/post.hpp>

#include "backend/backend.h"
#include "base/asio_error.h"
#include "checkpoint/checkpoint.h"
#include "model/qwen35/qwen35_model.h"
#include "model/registry.h"
#include "runtime/precision.h"
#include "worker/model_runner.h"

namespace ccinfer {
Worker::BatchTranslator::BatchTranslator(Backend& backend, int block_size)
    : backend_(backend), block_size_(block_size) {}

Result<PhysicalBatch> Worker::BatchTranslator::translate(const ScheduledBatch& batch) const {
    assert(block_size_ > 0);
    assert(!batch.items.empty());

    const std::size_t count = batch.items.size();
    int max_blocks = 0;
    int total_tokens = 0;
    for (const auto& item : batch.items) {
        const int context = std::visit(
            [](const auto& work) { return work.expected_context_len.value_or(0); }, item);
        const BlockTable* table =
            std::visit([](const auto& work) { return &work.block_table; }, item);
        const bool writes =
            std::holds_alternative<PrefillChunk>(item) || std::get<DecodeOneToken>(item).write_kv;
        if (const auto* decode = std::get_if<DecodeOneToken>(&item); decode)
            assert(!decode->late_bind);
        assert(!table->empty());
        assert(context >= 0);
        assert(context + (writes ? work_token_count(item) : 0) <=
               table->token_capacity(block_size_));
        max_blocks = std::max(max_blocks, table->size());
        total_tokens += work_token_count(item);
    }

    std::vector<int32_t> token_ids(static_cast<std::size_t>(total_tokens));
    std::vector<int32_t> positions(static_cast<std::size_t>(total_tokens));
    std::vector<int32_t> slots(static_cast<std::size_t>(total_tokens));
    std::vector<int32_t> block_table(count * static_cast<std::size_t>(max_blocks), -1);
    std::vector<int32_t> query_start(count + 1);
    std::vector<int32_t> context_lens(count);
    std::vector<int32_t> logits_indices(count, -1);
    std::vector<int32_t> item_counts(count);
    std::vector<SequenceId> item_seq_ids(count);
    std::vector<WorkKind> item_kinds(count);
    std::vector<bool> sample_flags(count);
    std::vector<int32_t> logits_rows;
    int offset = 0;
    int num_logits = 0;
    int max_position = 0;
    bool has_prefill = false;
    bool has_decode = false;

    for (std::size_t i = 0; i < count; ++i) {
        const auto& item = batch.items[i];
        const int context = std::visit(
            [](const auto& work) { return work.expected_context_len.value_or(0); }, item);
        const auto& table = *std::visit([](const auto& work) { return &work.block_table; }, item);
        const bool prefill = std::holds_alternative<PrefillChunk>(item);
        const bool writes = prefill || std::get<DecodeOneToken>(item).write_kv;
        const int tokens = work_token_count(item);
        item_counts[i] = tokens;
        item_seq_ids[i] = work_sequence_id(item);
        item_kinds[i] = work_kind(item);
        has_prefill |= prefill;
        has_decode |= !prefill;
        query_start[i] = offset;
        for (int b = 0; b < table.size(); ++b)
            block_table[i * static_cast<std::size_t>(max_blocks) + static_cast<std::size_t>(b)] =
                table[b];

        std::visit(
            [&](const auto& work) {
                using T = std::decay_t<decltype(work)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    for (int t = 0; t < tokens; ++t) {
                        const int pos = context + t;
                        token_ids[static_cast<std::size_t>(offset + t)] = work.tokens[t];
                        positions[static_cast<std::size_t>(offset + t)] = pos;
                        slots[static_cast<std::size_t>(offset + t)] =
                            table[pos / block_size_] * block_size_ + pos % block_size_;
                    }
                    sample_flags[i] = work.needs_sample;
                } else {
                    const int pos = writes ? context : std::max(0, context - 1);
                    token_ids[static_cast<std::size_t>(offset)] = work.input_token;
                    positions[static_cast<std::size_t>(offset)] = pos;
                    slots[static_cast<std::size_t>(offset)] =
                        writes ? table[pos / block_size_] * block_size_ + pos % block_size_ : -1;
                    sample_flags[i] = true;
                }
            },
            item);
        context_lens[i] = context + (writes ? tokens : 0);
        max_position =
            std::max(max_position, positions[static_cast<std::size_t>(offset + tokens - 1)]);
        if (sample_flags[i]) {
            logits_indices[i] = num_logits++;
            logits_rows.push_back(offset + tokens - 1);
        }
        offset += tokens;
    }
    query_start[count] = offset;

    PhysicalBatch result;
    result.num_tokens = total_tokens;
    result.batch_size = static_cast<int>(count);
    result.max_blocks_per_req = max_blocks;
    result.mode = has_prefill && has_decode ? ForwardMode::Mixed
                  : has_prefill             ? ForwardMode::Prefill
                                            : ForwardMode::Decode;
    result.item_seq_ids = std::move(item_seq_ids);
    result.item_kinds = std::move(item_kinds);
    result.item_token_counts = std::move(item_counts);
    result.sample_flags = std::move(sample_flags);
    result.logits_rows_host = std::move(logits_rows);
    result.num_logits = num_logits;
    result.max_position_id = max_position;

    auto upload = [&](const std::vector<int32_t>& values,
                      std::initializer_list<std::int64_t> shape) -> Result<Tensor> {
        return Tensor::from_host(backend_, values.data(), ccop::DType::kInt32, shape);
    };
    auto token_r = upload(token_ids, {total_tokens});
    if (!token_r) return std::unexpected(token_r.error());
    result.token_ids = std::move(*token_r);
    auto position_r = upload(positions, {total_tokens});
    if (!position_r) return std::unexpected(position_r.error());
    result.positions = std::move(*position_r);
    auto slot_r = upload(slots, {total_tokens});
    if (!slot_r) return std::unexpected(slot_r.error());
    result.slot_mapping = std::move(*slot_r);
    auto table_r = upload(block_table, {static_cast<std::int64_t>(count), max_blocks});
    if (!table_r) return std::unexpected(table_r.error());
    result.block_table = std::move(*table_r);
    auto query_r = upload(query_start, {static_cast<std::int64_t>(count + 1)});
    if (!query_r) return std::unexpected(query_r.error());
    result.query_start_loc = std::move(*query_r);
    auto context_r = upload(context_lens, {static_cast<std::int64_t>(count)});
    if (!context_r) return std::unexpected(context_r.error());
    result.context_lens = std::move(*context_r);
    auto logits_r = upload(logits_indices, {static_cast<std::int64_t>(count)});
    if (!logits_r) return std::unexpected(logits_r.error());
    result.logits_indices = std::move(*logits_r);
    auto sync_r = backend_.synchronize();
    if (!sync_r) return std::unexpected(sync_r.error());
    return result;
}

Worker::Worker(asio::io_context& io) : io_(io) {}

Worker::~Worker() { shutdown(); }

Result<void> Worker::init(const std::string& model_path, const ModelConfig& model,
                          const EngineConfig& engine) {
    if (running_.load() || worker_thread_.joinable())
        return std::unexpected(ErrorCode::InvalidArgument);
    auto result = init_resources(model_path, model, engine);
    if (!result) {
        reset_resources();
        return result;
    }
    running_.store(true);
    worker_thread_ = std::thread(&Worker::worker_loop, this);
    return {};
}

void Worker::shutdown() {
    running_.store(false);
    cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

Result<BatchFuture> Worker::enqueue_execute_batch(ScheduledBatch batch) {
    auto channel = std::make_shared<BatchChannel>(io_.get_executor(), 1);
    std::unique_lock lock(queue_mutex_);
    if (!running_) return std::unexpected(ErrorCode::ServerShuttingDown);
    queue_.push_back(PendingBatch{std::move(batch), channel});
    lock.unlock();
    cv_.notify_one();
    return channel;
}

void Worker::resolve(const std::shared_ptr<BatchChannel>& channel, Result<BatchResult> result) {
    asio::post(io_, [channel, result = std::move(result)]() mutable {
        channel->async_send(
            boost::system::error_code{}, std::move(result), [](boost::system::error_code ec) {
                if (ec) {
                    ccLog::warn("worker completion channel failed ec={}", ec.value());
                }
            });
    });
}

void Worker::worker_loop() {
    while (true) {
        std::deque<PendingBatch> local_queue;
        {
            std::unique_lock lock(queue_mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
            if (!running_ && queue_.empty()) break;
            local_queue = std::move(queue_);
            queue_.clear();
        }
        for (auto& command : local_queue) process_command(std::move(command));
    }
    reset_resources();
}

void Worker::process_command(PendingBatch pending) {
    if (!running_.load()) {
        resolve(pending.chan, Result<BatchResult>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }
    process_batch(std::move(pending));
}

Worker::ResolvedBatch Worker::resolve_batch(const ScheduledBatch& batch) const {
    ResolvedBatch resolved;
    resolved.batch.batch_id = batch.batch_id;
    resolved.batch.sampling = batch.sampling;
    resolved.batch.items.reserve(batch.items.size());
    resolved.original_indices.reserve(batch.items.size());

    for (std::size_t i = 0; i < batch.items.size(); ++i) {
        WorkItem item = batch.items[i];
        const SequenceId seq_id = work_sequence_id(item);
        bool stale = failed_sequences_.contains(seq_id);
        std::visit(
            [&](auto& work) {
                if constexpr (std::is_same_v<std::decay_t<decltype(work)>, DecodeOneToken>) {
                    if (stale || !work.late_bind) return;
                    // latest_tokens_ persists across batches and preemption:
                    // the token may come from any earlier completed batch,
                    // not necessarily the immediately preceding one.
                    auto token = latest_tokens_.find(seq_id);
                    assert(token != latest_tokens_.end() &&
                           "late-bound token must have been produced by an earlier batch");
                    work.input_token = token->second;
                    work.late_bind = false;
                }
            },
            item);
        if (stale) {
            WorkItemResult result;
            result.item_index = static_cast<int>(i);
            result.seq_id = seq_id;
            result.kind = work_kind(batch.items[i]);
            result.stale = true;
            resolved.stale_results.push_back(std::move(result));
            continue;
        }
        resolved.batch.items.push_back(std::move(item));
        resolved.original_indices.push_back(i);
    }
    return resolved;
}

void Worker::process_batch(PendingBatch pending) {
    assert(initialized_ && backend_ && block_storage_);
    assert(!pending.batch.items.empty());
    auto resolved = resolve_batch(pending.batch);
    if (resolved.batch.items.empty()) {
        BatchResult result;
        result.batch_id = pending.batch.batch_id;
        result.items = std::move(resolved.stale_results);
        resolve(pending.chan, Result<BatchResult>(std::move(result)));
        return;
    }

    assert(model_);
    auto fail_sequences = [&] {
        for (const auto& item : resolved.batch.items)
            failed_sequences_.insert(work_sequence_id(item));
    };
    BatchTranslator translator(*backend_, engine_config_.kv_block_size);
    auto physical = translator.translate(resolved.batch);
    if (!physical) {
        fail_sequences();
        resolve(pending.chan, Result<BatchResult>(std::unexpected(physical.error())));
        return;
    }

    BatchResult result;
    result.batch_id = pending.batch.batch_id;
    auto execution = ModelRunner::inference<Qwen3ExecutionTraits>(
        *model_, *physical, *backend_, *block_storage_, resolved.batch.sampling);
    if (!execution) {
        fail_sequences();
        resolve(pending.chan, Result<BatchResult>(std::unexpected(execution.error())));
        return;
    }
    result.items = std::move(*execution);
    for (auto& item_result : result.items) {
        const std::size_t index = static_cast<std::size_t>(item_result.item_index);
        item_result.item_index = static_cast<int>(resolved.original_indices[index]);
        const auto& item = pending.batch.items[static_cast<std::size_t>(item_result.item_index)];
        if (!item_result.sampled_tokens.empty()) {
            latest_tokens_[item_result.seq_id] = item_result.sampled_tokens.front();
        } else {
            std::visit(
                [&](const auto& work) {
                    using T = std::decay_t<decltype(work)>;
                    if constexpr (std::is_same_v<T, PrefillChunk>) {
                        if (!work.tokens.empty())
                            latest_tokens_[item_result.seq_id] = work.tokens.back();
                    } else if (work.write_kv) {
                        latest_tokens_[item_result.seq_id] = work.input_token;
                    }
                },
                item);
        }
    }
    result.items.insert(result.items.end(), std::make_move_iterator(resolved.stale_results.begin()),
                        std::make_move_iterator(resolved.stale_results.end()));
    std::sort(result.items.begin(), result.items.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.item_index < rhs.item_index; });

    resolve(pending.chan, Result<BatchResult>(std::move(result)));
}

Result<void> Worker::init_resources(const std::string& model_path, const ModelConfig& model,
                                    const EngineConfig& engine) {
    try {
        engine_config_ = engine;
        auto backend = Backend::create(engine_config_.device_id);
        if (!backend) return std::unexpected(backend.error());
        backend_ = std::move(*backend);
        static std::once_flag register_flag;
        std::call_once(register_flag, register_builtin_models);
        auto checkpoint = Checkpoint::open(model_path);
        if (!checkpoint) return std::unexpected(checkpoint.error());
        auto model_handle =
            ModelRegistry::instance().create(model, (*checkpoint)->weights(), *backend_);
        if (!model_handle) return std::unexpected(model_handle.error());
        model_ = std::move(*model_handle);
        const int num_kv_layers =
            model.arch_ == ModelArch::Qwen3_5 ? qwen35::num_kv_layers(model) : model.n_layers_;
        auto storage = BlockStorage::create(*backend_, num_kv_layers, engine_config_.max_blocks,
                                            engine_config_.kv_block_size, model.n_kv_heads_,
                                            model.head_dim_, ccop::DType::kBFloat16);
        if (!storage) return std::unexpected(storage.error());
        block_storage_ = std::move(*storage);
        assert(block_storage_->block_size() == engine_config_.kv_block_size);
        initialized_ = true;
        return {};
    } catch (...) {
        return std::unexpected(ErrorCode::InternalError);
    }
}

void Worker::reset_resources() {
    initialized_ = false;
    latest_tokens_.clear();
    failed_sequences_.clear();
    model_.reset();
    block_storage_.reset();
    backend_.reset();
}

}  // namespace ccinfer
