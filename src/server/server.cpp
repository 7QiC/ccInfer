#include "server/server.h"

#include <atomic>
#include <future>
#include <iostream>
#include <memory>

#include "common/error_code.h"
#include "engine/executor/executor.h"
#include "server/http/http_server.h"
#include "server/scheduler/scheduler.h"
#include "server/tokenizer/tokenizer.h"

namespace ccinfer {
namespace server {

struct Server::Impl {
    asio::io_context& http_io;
    asio::io_context& scheduler_io;
    engine::Executor& executor;
    uint16_t port;
    std::string model_dir;

    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<Scheduler> scheduler;
    std::unique_ptr<HttpServer> http;

    std::shared_future<void> http_shutdown_future;
    std::shared_future<void> scheduler_shutdown_future;
    std::atomic<bool> shutdown_started{false};
    std::atomic<bool> started{false};

    Impl(asio::io_context& hi, asio::io_context& si, engine::Executor& e, uint16_t p,
         const std::string& md)
        : http_io(hi), scheduler_io(si), executor(e), port(p), model_dir(md) {}
};

Server::Server(asio::io_context& http_io, asio::io_context& scheduler_io,
               engine::Executor& executor,
               uint16_t port, const std::string& model_dir)
    : impl_(std::make_unique<Impl>(http_io, scheduler_io, executor, port, model_dir)) {}

// Owner must call shutdown() + wait_shutdown() before destroying Server.
Server::~Server() = default;

Result<void> Server::start() {
    if (impl_->shutdown_started.load()) {
        return std::unexpected(ErrorCode::ServerShuttingDown);
    }
    if (impl_->started.exchange(true)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto tok_r = create_tokenizer(impl_->model_dir);
    if (!tok_r) return std::unexpected(tok_r.error());
    impl_->tokenizer = std::move(*tok_r);

    impl_->scheduler = std::make_unique<Scheduler>(impl_->scheduler_io, impl_->executor);
    impl_->scheduler->start();

    impl_->http = std::make_unique<HttpServer>(impl_->http_io, impl_->port, *impl_->scheduler,
                                               *impl_->tokenizer);
    auto http_result = impl_->http->start();
    if (!http_result) return std::unexpected(http_result.error());

    std::cout << "Server listening on port " << impl_->port << std::endl;
    return {};
}

void Server::shutdown() {
    if (impl_->shutdown_started.exchange(true)) return;

    // Shut down HTTP first: closes sockets and channels, unblocking SSE
    // coroutines so they stop posting scheduler.cancel() callbacks.
    if (impl_->http) {
        impl_->http_shutdown_future = impl_->http->shutdown_async();
    }
}

void Server::wait_shutdown() {
    // Wait for HTTP drain to complete — all sockets/channels closed,
    // all SSE coroutines exited, no more scheduler.cancel() callbacks.
    if (impl_->http_shutdown_future.valid()) {
        impl_->http_shutdown_future.wait();
    }

    // Now safe to shut down Scheduler: no HTTP callbacks will reference it.
    if (impl_->scheduler) {
        impl_->scheduler_shutdown_future = impl_->scheduler->shutdown_async();
    }
    if (impl_->scheduler_shutdown_future.valid()) {
        impl_->scheduler_shutdown_future.wait();
    }
}

}  // namespace server
}  // namespace ccinfer
