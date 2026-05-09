#pragma once

#include <atomic>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/streambuf.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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
// Phase 4.1: RequestState access assumes io_context single-threaded run.
// If run in multiple threads, RequestState fields need strand/atomic/lock.

class HttpServer {
public:
    HttpServer(asio::io_context& io, uint16_t port, Scheduler& scheduler);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void start();
    void shutdown();

private:
    asio::awaitable<void> accept_loop();
    asio::awaitable<void> handle_connection(asio::ip::tcp::socket socket);

    asio::awaitable<std::string> read_line(asio::ip::tcp::socket& socket, asio::streambuf& buf);
    asio::awaitable<std::optional<std::string>> read_body(asio::ip::tcp::socket& socket,
                                                          asio::streambuf& buf,
                                                          size_t content_length);
    asio::awaitable<void> write_response(asio::ip::tcp::socket& socket, std::string_view response);

    asio::awaitable<void> handle_health(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_models(asio::ip::tcp::socket& socket);
    asio::awaitable<void> handle_chat(asio::ip::tcp::socket& socket, std::string body);

    asio::io_context& io_;
    uint16_t port_;
    Scheduler& scheduler_;
    asio::ip::tcp::acceptor acceptor_;
    std::atomic<bool> running_{false};
};

}  // namespace server
}  // namespace ccinfer
