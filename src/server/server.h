#pragma once

#include <boost/asio/io_context.hpp>
#include <cstdint>
#include <memory>

namespace ccinfer {

namespace asio = boost::asio;

namespace engine {
class Engine;
}

namespace server {

class Server {
public:
    Server(asio::io_context& http_io, asio::io_context& scheduler_io,
           engine::Engine& engine, uint16_t port);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    void start();

    // Initiates shutdown of HttpServer and Scheduler.
    // Does NOT stop io_contexts, shutdown Engine, or destroy objects.
    void shutdown();

    // Blocks until both HttpServer and Scheduler have completed shutdown.
    // Must be called after shutdown() and before engine.shutdown().
    void wait_shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace server
}  // namespace ccinfer
