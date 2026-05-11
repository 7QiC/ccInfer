#include <atomic>
#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "common/error_code.h"
#include "engine/engine.h"
#include "server/server.h"

int main(int argc, char* argv[]) {
    // Parse args
    int port = 8080;
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (port <= 0 || port > 65535) {
        std::cerr << "Invalid port: " << port << std::endl;
        return 1;
    }
    if (model_path.empty()) {
        std::cerr << "Error: --model-path is required (no dummy tokenizer available)" << std::endl;
        return 1;
    }

    // Infrastructure — work guards prevent run() from returning early
    // when there is temporarily no work.
    boost::asio::io_context http_io;
    boost::asio::io_context scheduler_io;
    auto http_guard = boost::asio::make_work_guard(http_io);
    auto scheduler_guard = boost::asio::make_work_guard(scheduler_io);

    // Engine (engine layer)
    ccinfer::engine::Engine engine(scheduler_io);
    if (auto r = engine.init(model_path); !r) {
        std::cerr << "Engine init failed: " << ccinfer::error_message(r.error()) << std::endl;
        return 1;
    }

    // Server (server layer)
    ccinfer::server::Server server(http_io, scheduler_io, engine, static_cast<uint16_t>(port),
                                   model_path);
    auto sr = server.start();
    if (!sr) {
        std::cerr << "Server start failed: " << ccinfer::error_message(sr.error()) << std::endl;
        engine.shutdown();
        return 1;
    }

    // Signal handling — only sets a flag; no complex work in signal handler.
    std::atomic<bool> shutdown_flag{false};
    boost::asio::signal_set signals(http_io, SIGINT, SIGTERM);
    signals.async_wait([&shutdown_flag](const boost::system::error_code& ec, int) {
        if (!ec) shutdown_flag.store(true);
    });

    // Threads
    std::thread http_thread([&http_io] { http_io.run(); });
    std::thread sched_thread([&scheduler_io] { scheduler_io.run(); });

    // Wait for shutdown signal
    while (!shutdown_flag.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Graceful shutdown.
    //   1. server.shutdown() — triggers HttpServer shutdown (closes sockets,
    //      channels, drains SSE coroutines).
    //   2. server.wait_shutdown() — waits for HTTP drain, then triggers and
    //      waits for Scheduler cleanup (releases sequences, sends terminal
    //      events through already-draining channels).
    //   3. engine.shutdown() — all sequences released, safe to stop Engine.
    //   4. Reset work guards + stop io_contexts + join threads.
    std::cout << "Shutting down..." << std::endl;

    server.shutdown();
    server.wait_shutdown();

    engine.shutdown();

    http_guard.reset();
    scheduler_guard.reset();

    scheduler_io.stop();
    http_io.stop();

    if (http_thread.joinable()) http_thread.join();
    if (sched_thread.joinable()) sched_thread.join();

    std::cout << "Server stopped." << std::endl;
    return 0;
}
