#include "executor/single_device_executor.h"

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/this_coro.hpp>
#include <unordered_set>
#include <utility>
#include <variant>

#include "base/asio_error.h"
#include "base/channel.h"
#include "worker/worker.h"

namespace ccinfer {

namespace {
using asio::as_tuple;
using asio::deferred;
}  // namespace

SingleDeviceExecutor::SingleDeviceExecutor(boost::asio::io_context& io)
    : worker_(std::make_unique<Worker>(io)) {}

SingleDeviceExecutor::~SingleDeviceExecutor() { shutdown(); }

Result<void> SingleDeviceExecutor::init(const std::string& model_path) {
    return worker_->init(model_path);
}

void SingleDeviceExecutor::shutdown() { worker_->shutdown(); }

asio::awaitable<Result<CreateSequenceResult>> SingleDeviceExecutor::create_sequence(
    std::vector<int32_t> prompt_tokens, int max_context_len) {
    SequenceId seq_id = next_seq_id_++;
    auto prompt_copy = prompt_tokens;

    auto result = worker_->prepare_sequence_resources(seq_id, prompt_tokens, max_context_len);
    if (!result) co_return std::unexpected(result.error());
    if (result->seq_id != seq_id) co_return std::unexpected(ErrorCode::InternalError);

    SequenceSnapshot state;
    state.seq_id = seq_id;
    state.prompt_tokens = std::move(prompt_copy);
    state.max_context_len = max_context_len;
    state.kv_written = result->prompt_processed;
    state.prompt_processed = result->prompt_processed;
    sequences_[seq_id] = std::move(state);
    co_return result;
}

asio::awaitable<Result<SuspendSequenceResult>> SingleDeviceExecutor::suspend_sequence(
    SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) co_return std::unexpected(ErrorCode::InvalidArgument);
    auto prompt_copy = prompt_tokens;

    auto result = worker_->reset_sequence_resources(seq_id, prompt_tokens, max_context_len);
    if (!result) co_return std::unexpected(result.error());

    auto& state = it->second;
    state.prompt_tokens = std::move(prompt_copy);
    state.max_context_len = max_context_len;
    state.kv_written = result->prompt_processed;
    state.prompt_processed = result->prompt_processed;
    state.aborted = false;
    co_return result;
}

asio::awaitable<Result<void>> SingleDeviceExecutor::release_sequence(SequenceId seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) co_return Result<void>{};

    auto result = worker_->release_sequence_resources(seq_id);
    if (!result) co_return std::unexpected(result.error());
    sequences_.erase(it);
    co_return result;
}

asio::awaitable<Result<void>> SingleDeviceExecutor::abort_sequence(SequenceId seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) co_return Result<void>{};

    auto result = worker_->release_sequence_resources(seq_id);
    if (!result) co_return std::unexpected(result.error());
    sequences_.erase(it);
    co_return result;
}

asio::awaitable<Result<BatchResult>> SingleDeviceExecutor::execute_batch(ScheduledBatch batch) {
    std::unordered_set<SequenceId> seen;
    std::vector<SequenceSnapshot> snapshots;
    snapshots.reserve(batch.items.size());

    for (const auto& item : batch.items) {
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        if (!seen.insert(seq_id).second) continue;

        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) co_return std::unexpected(ErrorCode::InvalidArgument);
        snapshots.push_back(it->second);
    }

    auto chan = std::make_shared<BatchChannel>(co_await asio::this_coro::executor, 1);
    worker_->enqueue_execute_batch(std::move(batch), std::move(snapshots), chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());

    for (const auto& delta : result->deltas) {
        auto it = sequences_.find(delta.seq_id);
        if (it == sequences_.end()) co_return std::unexpected(ErrorCode::InternalError);
        auto& state = it->second;
        if (delta.kv_tokens_committed < 0 || delta.prompt_tokens_committed < 0) {
            co_return std::unexpected(ErrorCode::InternalError);
        }
        state.kv_written += delta.kv_tokens_committed;
        state.prompt_processed += delta.prompt_tokens_committed;
        if (state.kv_written > state.max_context_len ||
            state.prompt_processed > static_cast<int>(state.prompt_tokens.size()) ||
            state.prompt_processed > state.kv_written) {
            co_return std::unexpected(ErrorCode::InternalError);
        }
    }

    co_return std::move(result->batch);
}

EngineCapacity SingleDeviceExecutor::capacity() const {
    auto cap = worker_->capacity();
    return EngineCapacity{cap.max_sequences, cap.active_sequences, cap.free_blocks, cap.max_blocks,
                          cap.block_size, cap.block_active, cap.block_cached_idle,
                          cap.prefix_lookup_hits, cap.prefix_lookup_misses,
                          cap.prefix_evictions, cap.prefix_cached_blocks};
}

std::unique_ptr<Executor> Executor::create(boost::asio::io_context& io) {
    return std::make_unique<SingleDeviceExecutor>(io);
}

}  // namespace ccinfer
