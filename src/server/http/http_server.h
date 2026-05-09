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
#include <optional>
#include <set>
#include <string>
#include <string_view>

namespace ccinfer {
namespace server {

namespace asio = boost::asio;

class Scheduler;

// HttpServer must outlive detached accept_loop/connection coroutines.
// Server shutdown must call shutdown() and keep io_context running
// until outstanding coroutines exit.
//
// ConnectionState is only accessed on the http_io thread.
// Scheduler communication goes through submit()/cancel() which post
// to the scheduler_io thread — no shared mutable state across threads.

class HttpServer {
public:
    HttpServer(asio::io_context& io, uint16_t port, Scheduler& scheduler);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void start();

    // Fire-and-forget shutdown.  Prefer shutdown_async() for waitable shutdown.
    void shutdown();

    // Initiates shutdown; returns a future that is set when the accept loop
    // has exited and all active sockets have been cleaned up.
    std::shared_future<void> shutdown_async();

private:
    asio::awaitable<void> accept_loop();
    asio::awaitable<void> accept_loop_impl();
    asio::awaitable<void> handle_connection(std::shared_ptr<asio::ip::tcp::socket> socket);
    asio::awaitable<void> handle_connection_impl(asio::ip::tcp::socket& socket);

    asio::awaitable<std::string> read_line(asio::ip::tcp::socket& socket, asio::streambuf& buf);
    asio::awaitable<std::optional<std::string>> read_body(asio::ip::tcp::socket& socket,
                                                          asio::streambuf& buf,
                                                          size_t content_length);
    asio::awaitable<void> write_response(asio::ip::tcp::socket& socket, std::string_view response);

    asio::awaitable<void> handle_health(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_models(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_chat(asio::ip::tcp::socket& socket, std::string body);

    void try_finish_shutdown_on_http_thread();

    asio::io_context& io_;
    uint16_t port_;
    Scheduler& scheduler_;
    asio::ip::tcp::acceptor acceptor_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_request_id_{1};

    // Only accessed on http_io thread.
    std::set<std::shared_ptr<asio::ip::tcp::socket>> active_sockets_;
    bool accept_loop_done_{false};

    // Shutdown synchronisation.
    std::unique_ptr<std::promise<void>> shutdown_promise_;
    std::shared_future<void> shutdown_future_;
    bool shutdown_done_sent_{false};
};

}  // namespace server
}  // namespace ccinfer
