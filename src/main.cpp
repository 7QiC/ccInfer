#include <atomic>
#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>

#include "base/error_code.h"
#include "base/runtime_config.h"
#include "executor/executor.h"
#include "http/http_server.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"

namespace {

bool parse_nonnegative_int(const char* text, int& value) {
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (*end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --model-path PATH [--port PORT] [--prefill-chunk-size N]\n";
}

}  // namespace

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
        } else if (arg == "--prefill-chunk-size" && i + 1 < argc) {
            int chunk_size = 0;
            if (!parse_nonnegative_int(argv[++i], chunk_size)) {
                std::cerr << "Invalid prefill chunk size: " << argv[i] << std::endl;
                return 1;
            }
            ccinfer::runtime::set_prefill_chunk_size(chunk_size);
        } else {
            print_usage(argv[0]);
            return 1;
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

    // Executor (runtime layer)
    auto executor = ccinfer::Executor::create(scheduler_io);
    if (auto r = executor->init(model_path); !r) {
        std::cerr << "Executor init failed: " << ccinfer::error_message(r.error()) << std::endl;
        return 1;
    }

    auto tok_r = ccinfer::create_tokenizer(model_path);
    if (!tok_r) {
        std::cerr << "Tokenizer init failed: " << ccinfer::error_message(tok_r.error())
                  << std::endl;
        executor->shutdown();
        return 1;
    }
    auto tokenizer = std::move(*tok_r);

    ccinfer::Scheduler scheduler(scheduler_io, *executor);
    scheduler.start();

    ccinfer::HttpServer http_server(http_io, static_cast<uint16_t>(port), scheduler, *tokenizer);
    auto sr = http_server.start();
    if (!sr) {
        std::cerr << "HTTP server start failed: " << ccinfer::error_message(sr.error())
                  << std::endl;
        auto scheduler_shutdown = scheduler.shutdown_async();
        if (scheduler_shutdown.valid()) scheduler_shutdown.wait();
        executor->shutdown();
        return 1;
    }
    std::cout << "Server listening on port " << port << std::endl;

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
    //   1. HttpServer shutdown closes sockets/channels and drains HTTP coroutines.
    //   2. Scheduler shutdown runs after HTTP drain, so no TokenSink callbacks can
    //      race with scheduler cleanup.
    //   3. Executor shutdown runs after all sequences have been released.
    //   4. Reset work guards + stop io_contexts + join threads.
    std::cout << "Shutting down..." << std::endl;

    auto http_shutdown = http_server.shutdown_async();
    if (http_shutdown.valid()) http_shutdown.wait();

    auto scheduler_shutdown = scheduler.shutdown_async();
    if (scheduler_shutdown.valid()) scheduler_shutdown.wait();

    executor->shutdown();

    http_guard.reset();
    scheduler_guard.reset();

    scheduler_io.stop();
    http_io.stop();

    if (http_thread.joinable()) http_thread.join();
    if (sched_thread.joinable()) sched_thread.join();

    std::cout << "Server stopped." << std::endl;
    return 0;
}
