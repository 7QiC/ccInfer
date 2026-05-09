#include "engine/engine.h"

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/this_coro.hpp>
#include <utility>

#include "engine/executor/executor.h"

namespace ccinfer {
namespace engine {

namespace {
using asio::as_tuple;
using asio::deferred;
}  // namespace

Engine::Engine(asio::io_context& io) : executor_(Executor::create(io)) {}

Engine::~Engine() { shutdown(); }

Result<void> Engine::init(const std::string& model_path) { return executor_->init(model_path); }

void Engine::shutdown() { executor_->shutdown(); }

EngineCapacity Engine::capacity() const { return executor_->capacity(); }

asio::awaitable<Result<SequenceId>> Engine::create_sequence(std::vector<int32_t> prompt_tokens,
                                                            int max_context_len) {
    auto chan = std::make_shared<SeqIdChannel>(co_await asio::this_coro::executor, 1);
    executor_->enqueue_create_sequence(std::move(prompt_tokens), max_context_len, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<void>> Engine::release_sequence(SequenceId seq_id) {
    auto chan = std::make_shared<VoidChannel>(co_await asio::this_coro::executor, 1);
    executor_->enqueue_release_sequence(seq_id, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<void>> Engine::abort_sequence(SequenceId seq_id) {
    auto chan = std::make_shared<VoidChannel>(co_await asio::this_coro::executor, 1);
    executor_->enqueue_abort_sequence(seq_id, chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

asio::awaitable<Result<BatchResult>> Engine::execute_batch(ScheduledBatch batch) {
    auto chan = std::make_shared<BatchChannel>(co_await asio::this_coro::executor, 1);
    executor_->enqueue_execute_batch(std::move(batch), chan);

    auto [ec, result] = co_await chan->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    co_return result;
}

}  // namespace engine
}  // namespace ccinfer
