#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>

#include "base/error.h"
#include "config/engine_config.h"
#include "config/model_config.h"
#include "checkpoint/checkpoint.h"
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
              << " --model-path PATH [--port PORT] [--device N] [--max-blocks N] "
                 "[--kv-block-size N] [--max-sequences N] [--max-running-requests N] "
                 "[--max-concurrent-batches N] [--max-pending-requests N] "
                 "[--max-token-budget N] [--max-seq-prefill-tokens N] "
                 "[--max-context-len N]\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string model_path;
    ccinfer::EngineConfig engine_config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], port)) {
                std::cerr << "Invalid port: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.device_id)) {
                std::cerr << "Invalid device: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-blocks" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_blocks)) {
                std::cerr << "Invalid max blocks: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--kv-block-size" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.kv_block_size)) {
                std::cerr << "Invalid block size: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-sequences" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_sequences)) {
                std::cerr << "Invalid max sequences: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-running-requests" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_running_requests)) {
                std::cerr << "Invalid max running requests: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-concurrent-batches" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_concurrent_batches)) {
                std::cerr << "Invalid max concurrent batches: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-pending-requests" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_pending_requests)) {
                std::cerr << "Invalid max pending requests: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-token-budget" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_token_budget)) {
                std::cerr << "Invalid max token budget: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-seq-prefill-tokens" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.max_seq_prefill_tokens)) {
                std::cerr << "Invalid max sequence prefill tokens: " << argv[i] << std::endl;
                return 1;
            }
        } else if (arg == "--max-context-len" && i + 1 < argc) {
            if (!parse_nonnegative_int(argv[++i], engine_config.default_max_context_len)) {
                std::cerr << "Invalid max context length: " << argv[i] << std::endl;
                return 1;
            }
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
        std::cerr << "Error: --model-path is required" << std::endl;
        return 1;
    }

    auto engine_r = engine_config.validate();
    if (!engine_r) {
        std::cerr << "Invalid engine config: " << ccinfer::error_message(engine_r.error())
                  << std::endl;
        return 1;
    }

    auto checkpoint_r = ccinfer::Checkpoint::open(model_path);
    if (!checkpoint_r) {
        std::cerr << "Checkpoint open failed: " << ccinfer::error_message(checkpoint_r.error())
                  << std::endl;
        return 1;
    }
    auto model_r = (*checkpoint_r)->load_config();
    if (!model_r) {
        std::cerr << "Model config load failed: " << ccinfer::error_message(model_r.error())
                  << std::endl;
        return 1;
    }
    auto model_config = std::move(*model_r);

    // Infrastructure — work guards prevent run() from returning early
    // when there is temporarily no work.
    boost::asio::io_context http_io;
    boost::asio::io_context scheduler_io;
    auto http_guard = boost::asio::make_work_guard(http_io);
    auto scheduler_guard = boost::asio::make_work_guard(scheduler_io);

    auto executor = ccinfer::Executor::create(scheduler_io);
    if (auto r = executor->init(model_path, model_config, engine_config); !r) {
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

    ccinfer::Scheduler scheduler(scheduler_io, *executor, engine_config);
    scheduler.start();
    std::thread sched_thread([&scheduler_io] { scheduler_io.run(); });

    ccinfer::HttpServer http_server(http_io, static_cast<uint16_t>(port), scheduler, *tokenizer);
    auto sr = http_server.start();
    if (!sr) {
        std::cerr << "HTTP server start failed: " << ccinfer::error_message(sr.error())
                  << std::endl;
        auto scheduler_shutdown = scheduler.shutdown_async();
        if (scheduler_shutdown.valid()) scheduler_shutdown.wait();
        executor->shutdown();
        scheduler_guard.reset();
        scheduler_io.stop();
        if (sched_thread.joinable()) sched_thread.join();
        return 1;
    }
    std::cout << "Server listening on port " << port << std::endl;

    // Signal handling — only sets a flag; no complex work in signal handler.
    std::atomic<bool> shutdown_flag{false};
    boost::asio::signal_set signals(http_io, SIGINT, SIGTERM);
    signals.async_wait([&shutdown_flag](const boost::system::error_code& ec, int) {
        if (!ec) shutdown_flag.store(true);
    });

    std::thread http_thread([&http_io] { http_io.run(); });

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
