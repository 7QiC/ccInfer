#pragma once

#include <boost/asio/io_context.hpp>
#include <cstdint>
#include <memory>
#include <string>

#include "common/result.h"

namespace ccinfer {

namespace engine {
class Engine;
}

namespace server {

namespace asio = boost::asio;

class Tokenizer;

class Server {
public:
    Server(asio::io_context& http_io, asio::io_context& scheduler_io, engine::Engine& engine,
           uint16_t port, const std::string& model_dir);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // One-shot start.  Returns error if tokenizer fails or start is called
    // more than once.
    Result<void> start();

    // Initiates shutdown of HttpServer only (closes sockets, channels).
    // Scheduler shutdown is deferred to wait_shutdown() — invoked after
    // HTTP drain completes, ensuring Scheduler outlives all HTTP callbacks.
    // Does NOT stop io_contexts, shutdown Engine, or destroy objects.
    void shutdown();

    // Blocks until both HttpServer and Scheduler have completed shutdown.
    // Must be called after shutdown() and before engine.shutdown().
    // Must NOT be called from http_io or scheduler_io threads (would deadlock).
    //
    // Order: waits for HTTP drain (sockets/channels closed, SSE coroutines
    // exited), then triggers Scheduler shutdown and waits for it.
    //
    // Scheduler remains alive until HTTP drain completes so that any
    // already-posted TokenSink::on_send_failed callbacks can safely fire.
    // shutdown() / wait_shutdown() are intended for single-threaded control
    // (main thread only); concurrent calls require additional mutex guards.
    void wait_shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace server
}  // namespace ccinfer
