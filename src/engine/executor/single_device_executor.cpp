#include "engine/executor/single_device_executor.h"

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/this_coro.hpp>
#include <utility>

#include "common/asio_error.h"
#include "common/channel.h"
#include "engine/worker/worker.h"

namespace ccinfer {
namespace engine {

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
    auto chan = std::make_shared<SeqIdChannel>(co_await asio::this_coro::executor, 1);
    worker_->enqueue_create_sequence(std::move(prompt_tokens), max_context_len, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<SuspendSequenceResult>> SingleDeviceExecutor::suspend_sequence(
    SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len) {
    auto chan = std::make_shared<SuspendChannel>(co_await asio::this_coro::executor, 1);
    worker_->enqueue_suspend_sequence(seq_id, std::move(prompt_tokens), max_context_len, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<void>> SingleDeviceExecutor::release_sequence(SequenceId seq_id) {
    auto chan = std::make_shared<VoidChannel>(co_await asio::this_coro::executor, 1);
    worker_->enqueue_release_sequence(seq_id, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<void>> SingleDeviceExecutor::abort_sequence(SequenceId seq_id) {
    auto chan = std::make_shared<VoidChannel>(co_await asio::this_coro::executor, 1);
    worker_->enqueue_abort_sequence(seq_id, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<BatchResult>> SingleDeviceExecutor::execute_batch(ScheduledBatch batch) {
    auto chan = std::make_shared<BatchChannel>(co_await asio::this_coro::executor, 1);
    worker_->enqueue_execute_batch(std::move(batch), chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
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

}  // namespace engine
}  // namespace ccinfer
