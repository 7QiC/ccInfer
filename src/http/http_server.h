#pragma once

#include <atomic>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/streambuf.hpp>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>

#include "base/channel.h"
#include "base/result.h"

namespace ccinfer {

class Scheduler;
class Tokenizer;

namespace asio = boost::asio;

// HttpServer must outlive detached accept_loop/connection coroutines.
// Server shutdown must call shutdown() and keep io_context running
// until outstanding coroutines exit.
//
// All state is accessed only on the http_io thread except:
//   - running_ (atomic)
//   - shutdown_promise_ / shutdown_future_ (mutex protected)
//   - shutdown_done_sent_ (http_io thread only, set under mutex)

class HttpServer {
public:
    HttpServer(asio::io_context& io, uint16_t port, Scheduler& scheduler,
               Tokenizer& tokenizer);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    Result<void> start();

    // Fire-and-forget shutdown.  Prefer shutdown_async() for waitable shutdown.
    void shutdown();

    // Initiates shutdown; returns a future that is set when the accept loop
    // has exited and all active connections have been cleaned up.
    std::shared_future<void> shutdown_async();

private:
    struct ActiveConn {
        std::string request_id;
        std::shared_ptr<asio::ip::tcp::socket> socket;
        std::shared_ptr<TokenChannel> channel;
    };

    asio::awaitable<void> accept_loop();
    asio::awaitable<void> accept_loop_impl();
    asio::awaitable<void> handle_connection(std::shared_ptr<asio::ip::tcp::socket> socket,
                                            std::shared_ptr<ActiveConn> conn);
    asio::awaitable<void> handle_connection_impl(asio::ip::tcp::socket& socket,
                                                 std::shared_ptr<ActiveConn> conn);

    asio::awaitable<std::string> read_line(asio::ip::tcp::socket& socket, asio::streambuf& buf);
    asio::awaitable<std::optional<std::string>> read_body(asio::ip::tcp::socket& socket,
                                                          asio::streambuf& buf,
                                                          size_t content_length);
    asio::awaitable<void> write_response(asio::ip::tcp::socket& socket, std::string_view response);

    asio::awaitable<void> handle_health(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_models(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_metrics(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_chat(asio::ip::tcp::socket& socket, std::string body,
                                      const std::shared_ptr<ActiveConn>& conn);

    std::string make_sse_frame(const GeneratedToken& tok);

    void try_finish_shutdown_on_http_thread();

    asio::io_context& io_;
    uint16_t port_;
    Scheduler& scheduler_;
    Tokenizer& tokenizer_;
    asio::ip::tcp::acceptor acceptor_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_request_id_{1};

    // Only accessed on http_io thread.
    std::set<std::shared_ptr<ActiveConn>> active_conns_;
    bool accept_loop_done_{false};

    static constexpr size_t kMaxBodySize = 64 * 1024 * 1024;  // 64 MB
    static constexpr size_t kMaxHeaderSize = 128 * 1024;      // 128 KB

    // Shutdown synchronisation.
    // shutdown_promise_ / shutdown_future_ are protected by shutdown_mutex_
    // to serialise concurrent shutdown_async() calls on any thread.
    // shutdown_done_sent_ is only accessed on the http_io thread
    // (from try_finish_shutdown_on_http_thread).
    std::mutex shutdown_mutex_;
    std::unique_ptr<std::promise<void>> shutdown_promise_;
    std::shared_future<void> shutdown_future_;
    bool shutdown_done_sent_{false};  // http_io thread only
};

}  // namespace ccinfer
