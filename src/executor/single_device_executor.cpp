#include "executor/single_device_executor.h"

#include <unordered_set>
#include <utility>

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/deferred.hpp>

#include "common/asio_error.h"
#include "common/channel.h"
#include "worker/worker.h"

namespace ccinfer {

namespace {
using asio::as_tuple;
using asio::deferred;
}  // namespace

SingleDeviceExecutor::SingleDeviceExecutor(boost::asio::io_context& io)
    : worker_(std::make_unique<Worker>(io)) {}

SingleDeviceExecutor::~SingleDeviceExecutor() { shutdown(); }

Result<void> SingleDeviceExecutor::init(const Config& config) { return worker_->init(config); }

void SingleDeviceExecutor::shutdown() { worker_->shutdown(); }

asio::awaitable<Result<AdmitSequenceResult>> SingleDeviceExecutor::admit_sequence(
    std::vector<int32_t> prompt_tokens, int max_context_len, SequenceInitialState initial_state) {
    SequenceId seq_id = next_seq_id_++;

    auto channel_r =
        worker_->enqueue_admit_sequence(seq_id, prompt_tokens, max_context_len, initial_state);
    if (!channel_r) co_return std::unexpected(channel_r.error());
    auto [ec, result] = co_await (*channel_r)->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());
    if (result->seq_id != seq_id) co_return std::unexpected(ErrorCode::InternalError);

    SequenceSnapshot state;
    state.seq_id = seq_id;
    state.prompt_tokens = std::move(prompt_tokens);
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
    if (it->second.status == SequenceStatus::Aborted) {
        co_return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto channel_r = worker_->enqueue_suspend_sequence(seq_id, prompt_tokens, max_context_len);
    if (!channel_r) co_return std::unexpected(channel_r.error());
    auto [ec, result] = co_await (*channel_r)->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());

    auto& state = it->second;
    state.prompt_tokens = std::move(prompt_tokens);
    state.max_context_len = max_context_len;
    state.kv_written = result->prompt_processed;
    state.prompt_processed = result->prompt_processed;
    state.status = SequenceStatus::Suspended;
    co_return result;
}

asio::awaitable<Result<void>> SingleDeviceExecutor::release_sequence(SequenceId seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) co_return Result<void>{};

    auto channel_r = worker_->enqueue_release_sequence(seq_id);
    if (!channel_r) co_return std::unexpected(channel_r.error());
    auto [ec, result] = co_await (*channel_r)->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());
    sequences_.erase(it);
    co_return result;
}

asio::awaitable<Result<void>> SingleDeviceExecutor::abort_sequence(SequenceId seq_id) {
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) co_return Result<void>{};

    // Abort is terminal for this executor-side sequence. Keep the tombstone so
    // an already queued batch cannot apply deltas after the abort.
    it->second.status = SequenceStatus::Aborted;
    auto channel_r = worker_->enqueue_abort_sequence(seq_id);
    if (!channel_r) co_return std::unexpected(channel_r.error());
    auto [ec, result] = co_await (*channel_r)->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());
    co_return result;
}

Result<BatchFuture> SingleDeviceExecutor::execute_batch(ScheduledBatch batch) {
    std::unordered_set<SequenceId> seen;

    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        if (!seen.insert(seq_id).second) return std::unexpected(ErrorCode::InvalidArgument);

        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return std::unexpected(ErrorCode::InvalidArgument);
        if (it->second.status == SequenceStatus::Aborted) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
    }

    auto future = worker_->enqueue_execute_batch(std::move(batch));
    if (!future) return std::unexpected(future.error());
    return *future;
}

asio::awaitable<Result<BatchResult>> SingleDeviceExecutor::collect_batch(BatchFuture future) {
    if (!future) co_return std::unexpected(ErrorCode::InvalidArgument);
    auto [ec, result] = co_await future->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());

    for (const auto& delta : result->deltas) {
        auto it = sequences_.find(delta.seq_id);
        if (it == sequences_.end() || it->second.status == SequenceStatus::Aborted) {
            co_return std::unexpected(ErrorCode::InvalidArgument);
        }
    }

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
        state.status = SequenceStatus::Active;
    }

    co_return std::move(result->batch);
}

Capacity SingleDeviceExecutor::capacity() const { return worker_->capacity(); }

std::unique_ptr<Executor> Executor::create(boost::asio::io_context& io) {
    return std::make_unique<SingleDeviceExecutor>(io);
}

}  // namespace ccinfer
