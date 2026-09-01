#include "executor/single_device_executor.h"

#include <cassert>

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/deferred.hpp>

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

Result<void> SingleDeviceExecutor::init(const std::string& model_path, const ModelConfig& model,
                                        const EngineConfig& engine) {
    return worker_->init(model_path, model, engine);
}

void SingleDeviceExecutor::shutdown() { worker_->shutdown(); }

Result<BatchFuture> SingleDeviceExecutor::execute_batch(ScheduledBatch batch) {
    auto future = worker_->enqueue_execute_batch(std::move(batch));
    if (!future) return std::unexpected(future.error());
    return *future;
}

asio::awaitable<Result<BatchResult>> SingleDeviceExecutor::collect_batch(BatchFuture future) {
    assert(future);
    auto [ec, result] = co_await future->async_receive(as_tuple(deferred));
    if (ec) co_return std::unexpected(to_error_code(ec));
    if (!result) co_return std::unexpected(result.error());

    co_return std::move(*result);
}

std::unique_ptr<Executor> Executor::create(boost::asio::io_context& io) {
    return std::make_unique<SingleDeviceExecutor>(io);
}

}  // namespace ccinfer
