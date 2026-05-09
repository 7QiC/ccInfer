#include "server/server.h"

#include <atomic>
#include <future>
#include <iostream>
#include <memory>

#include "engine/engine.h"
#include "server/http/http_server.h"
#include "server/scheduler/scheduler.h"

namespace ccinfer {
namespace server {

struct Server::Impl {
    asio::io_context& http_io;
    asio::io_context& scheduler_io;
    engine::Engine& engine;
    uint16_t port;

    std::unique_ptr<Scheduler> scheduler;
    std::unique_ptr<HttpServer> http;

    std::shared_future<void> http_shutdown_future;
    std::shared_future<void> scheduler_shutdown_future;
    std::atomic<bool> shutdown_started{false};

    Impl(asio::io_context& hi, asio::io_context& si, engine::Engine& e, uint16_t p)
        : http_io(hi), scheduler_io(si), engine(e), port(p) {}
};

Server::Server(asio::io_context& http_io, asio::io_context& scheduler_io,
               engine::Engine& engine, uint16_t port)
    : impl_(std::make_unique<Impl>(http_io, scheduler_io, engine, port)) {}

// Owner must call shutdown() + wait_shutdown() before destroying Server.
// The destructor does NOT perform graceful shutdown — posting shutdown
// tasks that capture `this` and then immediately destroying the object
// would be a use-after-free.
Server::~Server() = default;

void Server::start() {
    impl_->scheduler = std::make_unique<Scheduler>(impl_->scheduler_io, impl_->engine);
    impl_->scheduler->start();

    impl_->http = std::make_unique<HttpServer>(impl_->http_io, impl_->port, *impl_->scheduler);
    impl_->http->start();

    std::cout << "Server listening on port " << impl_->port << std::endl;
}

void Server::shutdown() {
    if (impl_->shutdown_started.exchange(true)) return;

    if (impl_->http) {
        impl_->http_shutdown_future = impl_->http->shutdown_async();
    }
    if (impl_->scheduler) {
        impl_->scheduler_shutdown_future = impl_->scheduler->shutdown_async();
    }
}

void Server::wait_shutdown() {
    // Must be called from main/control thread only — NOT from http_io or
    // scheduler_io threads.  Blocking wait on an io thread would deadlock
    // because the shutdown completion work runs on those same threads.
    // Wait for Scheduler first: cleanup_all_active() sends terminal events
    // to HTTP TokenSinks, which helps HTTP coroutines drain.  Also, Engine
    // must not be shut down until scheduler cleanup releases all sequences.
    if (impl_->scheduler_shutdown_future.valid()) {
        impl_->scheduler_shutdown_future.wait();
    }

    if (impl_->http_shutdown_future.valid()) {
        impl_->http_shutdown_future.wait();
    }
}

}  // namespace server
}  // namespace ccinfer
