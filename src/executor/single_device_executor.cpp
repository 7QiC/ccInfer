#include "executor/single_device_executor.h"

#include <cassert>
#include <unordered_map>
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
    assert(result->seq_id == seq_id);

    SequenceSnapshot state;
    state.seq_id = seq_id;
    state.prompt_len = static_cast<int>(prompt_tokens.size());
    state.max_context_len = max_context_len;
    state.kv_written = result->prompt_processed;
    state.prompt_processed = result->prompt_processed;
    sequences_[seq_id] = std::move(state);
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

Result<BatchFuture> SingleDeviceExecutor::execute_batch(ScheduledBatch batch) {
#ifndef NDEBUG
    std::unordered_set<SequenceId> seen;
    for (const auto& item : batch.items) {
        assert(seen.insert(work_sequence_id(item)).second);
    }
#endif

    auto future = worker_->enqueue_execute_batch(std::move(batch));
    if (!future) return std::unexpected(future.error());
    return *future;
}

asio::awaitable<Result<BatchResult>> SingleDeviceExecutor::collect_batch(BatchFuture future) {
    assert(future);
    auto [ec, result] = co_await future->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());

    std::unordered_map<SequenceId, bool> eos_by_seq;
    for (const auto& item_result : result->batch.items) {
        if (item_result.eos) eos_by_seq[item_result.seq_id] = true;
    }

    for (const auto& delta : result->deltas) {
        auto it = sequences_.find(delta.seq_id);
        if (it == sequences_.end()) {
            continue;
        }

        auto& state = it->second;
        assert(delta.kv_tokens_committed >= 0 && delta.prompt_tokens_committed >= 0);
        state.kv_written += delta.kv_tokens_committed;
        state.prompt_processed += delta.prompt_tokens_committed;
        assert(state.kv_written <= state.max_context_len);
        assert(state.prompt_processed <= state.prompt_len);
        assert(state.prompt_processed <= state.kv_written);
        if (eos_by_seq[delta.seq_id]) state.finished = true;
    }

    co_return std::move(result->batch);
}

Capacity SingleDeviceExecutor::capacity() const { return worker_->capacity(); }

std::unique_ptr<Executor> Executor::create(boost::asio::io_context& io) {
    return std::make_unique<SingleDeviceExecutor>(io);
}

}  // namespace ccinfer
