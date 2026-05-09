#include "server/http/http_server.h"

#include <algorithm>
#include <boost/asio/as_tuple.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/write.hpp>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <istream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <utility>

#include "server/common/types.h"
#include "server/scheduler/scheduler.h"

namespace ccinfer {
namespace server {

namespace {
using asio::as_tuple;
using asio::deferred;
using asio::detached;

constexpr std::string_view kHealthResponse =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: 15\r\n"
    "Connection: close\r\n"
    "\r\n"
    "{\"status\":\"ok\"}";

constexpr std::string_view kModelsResponse =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: 27\r\n"
    "Connection: close\r\n"
    "\r\n"
    "{\"data\":[{\"id\":\"default\"}]}";

constexpr std::string_view kNotFoundResponse =
    "HTTP/1.1 404 Not Found\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n";

constexpr std::string_view kMethodNotAllowed =
    "HTTP/1.1 405 Method Not Allowed\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n";

constexpr std::string_view kBadRequest =
    "HTTP/1.1 400 Bad Request\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n";

// Phase 4.1 dummy tokenizer output.
std::string token_text(int32_t token_id) { return "tok_" + std::to_string(token_id); }

// Phase 4.1 internal SSE format, not OpenAI-compatible yet.
std::string sse_frame(const GeneratedToken& tok) {
    nlohmann::json j;
    if (tok.has_token) {
        j["token"] = token_text(tok.token_id);
        j["token_id"] = tok.token_id;
    }
    j["done"] = tok.finished;
    if (tok.error != ErrorCode::Ok) {
        j["error"] = std::string(error_message(tok.error));
    }
    return "data: " + j.dump() + "\n\n";
}

bool starts_with_ci(std::string_view s, std::string_view prefix) {
    if (s.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), s.begin(), [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) ==
               std::tolower(static_cast<unsigned char>(b));
    });
}

enum class CLState { NotPresent, Valid, Invalid };

struct CLResult {
    CLState state = CLState::NotPresent;
    size_t value = 0;
};

CLResult parse_content_length(std::string_view line) {
    if (!starts_with_ci(line, "content-length:")) {
        return {};
    }
    auto val = line.substr(15);  // len("content-length:")
    while (!val.empty() && std::isspace(static_cast<unsigned char>(val.front()))) {
        val.remove_prefix(1);
    }

    size_t out = 0;
    bool has_digit = false;
    size_t i = 0;
    for (; i < val.size(); ++i) {
        char ch = val[i];
        if (!std::isdigit(static_cast<unsigned char>(ch))) break;
        has_digit = true;
        out = out * 10 + static_cast<size_t>(ch - '0');
    }
    if (!has_digit) return {CLState::Invalid, 0};
    for (; i < val.size(); ++i) {
        if (!std::isspace(static_cast<unsigned char>(val[i]))) return {CLState::Invalid, 0};
    }
    return {CLState::Valid, out};
}

}  // namespace

HttpServer::HttpServer(asio::io_context& io, uint16_t port, Scheduler& scheduler)
    : io_(io), port_(port), scheduler_(scheduler), acceptor_(io) {}

HttpServer::~HttpServer() { shutdown(); }

void HttpServer::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    asio::co_spawn(io_, accept_loop(), detached);
}

void HttpServer::shutdown() {
    bool was_running = running_.exchange(false);
    if (!was_running) return;
    asio::post(io_, [this] {
        boost::system::error_code ec;
        acceptor_.cancel(ec);
        acceptor_.close(ec);
    });
    // TODO: close active sockets for clean shutdown.
}

// ---------------------------------------------------------------------------
// Connection accept
// ---------------------------------------------------------------------------

asio::awaitable<void> HttpServer::accept_loop() {
    boost::system::error_code ec;

    acceptor_.open(asio::ip::tcp::v4(), ec);
    if (ec) {
        running_ = false;
        co_return;
    }

    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true), ec);
    if (ec) {
        running_ = false;
        co_return;
    }

    acceptor_.bind(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port_), ec);
    if (ec) {
        running_ = false;
        co_return;
    }

    acceptor_.listen(asio::socket_base::max_listen_connections, ec);
    if (ec) {
        running_ = false;
        co_return;
    }

    while (running_) {
        auto [acc_ec, socket] = co_await acceptor_.async_accept(as_tuple(deferred));
        if (acc_ec) {
            if (!running_) break;
            continue;
        }
        asio::co_spawn(io_, handle_connection(std::move(socket)), detached);
    }
}

// ---------------------------------------------------------------------------
// HTTP parsing
// ---------------------------------------------------------------------------

asio::awaitable<std::string> HttpServer::read_line(asio::ip::tcp::socket& socket,
                                                   asio::streambuf& buf) {
    auto [ec, len] = co_await asio::async_read_until(socket, buf, "\r\n", as_tuple(deferred));
    if (ec) co_return "";

    std::istream is(&buf);
    std::string line;
    std::getline(is, line);

    if (!line.empty() && line.back() == '\r') line.pop_back();
    co_return line;
}

asio::awaitable<std::optional<std::string>> HttpServer::read_body(asio::ip::tcp::socket& socket,
                                                                  asio::streambuf& buf,
                                                                  size_t content_length) {
    std::string body;
    body.reserve(content_length);

    // Drain remaining bytes already in streambuf
    auto avail = buf.size();
    if (avail > 0) {
        size_t take = std::min(avail, content_length);
        body.resize(take);
        buf.sgetn(body.data(), static_cast<std::streamsize>(take));
    }

    // Read the rest from socket
    while (body.size() < content_length) {
        std::string chunk(std::min(content_length - body.size(), size_t{4096}), '\0');
        auto [ec, n] = co_await socket.async_read_some(asio::buffer(chunk.data(), chunk.size()),
                                                       as_tuple(deferred));
        if (ec || n == 0) co_return std::nullopt;
        body.append(chunk.data(), n);
    }

    co_return body;
}

asio::awaitable<void> HttpServer::write_response(asio::ip::tcp::socket& socket,
                                                 std::string_view response) {
    auto [ec, _] = co_await asio::async_write(socket, asio::buffer(response), as_tuple(deferred));
    (void)_;
    if (ec) co_return;
}

// ---------------------------------------------------------------------------
// Connection handler
// ---------------------------------------------------------------------------

asio::awaitable<void> HttpServer::handle_connection(asio::ip::tcp::socket socket) {
    asio::streambuf buf;

    auto request_line = co_await read_line(socket, buf);
    if (request_line.empty()) co_return;

    size_t sp1 = request_line.find(' ');
    size_t sp2 = request_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        co_await write_response(socket, kBadRequest);
        co_return;
    }

    std::string method = request_line.substr(0, sp1);
    std::string path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

    // Strip query string
    auto qpos = path.find('?');
    if (qpos != std::string::npos) path.resize(qpos);

    // Read headers
    size_t content_length = 0;
    bool has_content_length = false;
    bool content_length_malformed = false;
    while (true) {
        auto line = co_await read_line(socket, buf);
        if (line.empty()) break;
        auto cl = parse_content_length(line);
        if (cl.state == CLState::Invalid) {
            content_length_malformed = true;
            break;
        }
        if (cl.state == CLState::Valid) {
            if (has_content_length && content_length != cl.value) {
                content_length_malformed = true;
                break;
            }
            has_content_length = true;
            content_length = cl.value;
        }
    }
    if (content_length_malformed) {
        co_await write_response(socket, kBadRequest);
        co_return;
    }

    if (path == "/health" || path == "/healthz") {
        if (method != "GET") {
            co_await write_response(socket, kMethodNotAllowed);
            co_return;
        }
        co_await handle_health(socket);
    } else if (path == "/v1/models") {
        if (method != "GET") {
            co_await write_response(socket, kMethodNotAllowed);
            co_return;
        }
        co_await handle_models(socket);
    } else if (path == "/v1/chat/completions") {
        if (method != "POST") {
            co_await write_response(socket, kMethodNotAllowed);
            co_return;
        }
        auto body = co_await read_body(socket, buf, content_length);
        if (!body) {
            co_await write_response(socket, kBadRequest);
            co_return;
        }
        co_await handle_chat(socket, std::move(*body));
    } else {
        co_await write_response(socket, kNotFoundResponse);
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

asio::awaitable<void> HttpServer::handle_health(asio::ip::tcp::socket& socket) {
    co_await write_response(socket, kHealthResponse);
}

asio::awaitable<void> HttpServer::handle_models(asio::ip::tcp::socket& socket) {
    co_await write_response(socket, kModelsResponse);
}

asio::awaitable<void> HttpServer::handle_chat(asio::ip::tcp::socket& socket, std::string body) {
    // Parse JSON body
    nlohmann::json req_json = nlohmann::json::parse(body.empty() ? "{}" : body, nullptr, false);
    if (req_json.is_discarded()) {
        co_await write_response(socket, kBadRequest);
        co_return;
    }

    // Extract messages → tokens
    std::vector<int32_t> prompt_tokens;
    if (req_json.contains("messages") && req_json["messages"].is_array()) {
        for (const auto& msg : req_json["messages"]) {
            if (msg.contains("content") && msg["content"].is_string()) {
                std::string content = msg["content"].get<std::string>();
                for (char c : content) prompt_tokens.push_back(static_cast<int32_t>(c));
            }
        }
    }
    // Phase 4.1 dummy fallback when no messages provided.
    if (prompt_tokens.empty()) prompt_tokens = {1, 2, 3};

    // Sampling params with type checking and validation
    SamplingParams sampling;
    if (req_json.contains("max_tokens") && req_json["max_tokens"].is_number_integer()) {
        sampling.max_tokens = req_json["max_tokens"].get<int>();
    }
    if (req_json.contains("temperature") && req_json["temperature"].is_number()) {
        sampling.temperature = req_json["temperature"].get<float>();
    }
    if (req_json.contains("top_p") && req_json["top_p"].is_number()) {
        sampling.top_p = req_json["top_p"].get<float>();
    }
    if (req_json.contains("top_k") && req_json["top_k"].is_number_integer()) {
        sampling.top_k = req_json["top_k"].get<int>();
    }

    // Validate
    if (sampling.max_tokens < 0 || sampling.temperature < 0.0f || sampling.top_p <= 0.0f ||
        sampling.top_p > 1.0f || sampling.top_k < 0) {
        co_await write_response(socket, kBadRequest);
        co_return;
    }

    // Build request
    auto req = std::make_shared<RequestState>();
    req->request_id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    req->prompt_tokens = prompt_tokens;
    req->sampling = sampling;
    req->max_context_len = 2048;
    req->output_channel = std::make_shared<TokenChannel>(co_await asio::this_coro::executor, 16);

    scheduler_.enqueue(req);

    // Send SSE headers
    std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n";
    {
        auto [ec, _] =
            co_await asio::async_write(socket, asio::buffer(headers), as_tuple(deferred));
        if (ec) {
            req->cancelled = true;
            co_return;
        }
    }

    // Stream tokens
    while (true) {
        auto [recv_ec, tok] = co_await req->output_channel->async_receive(as_tuple(deferred));
        if (recv_ec) {
            req->cancelled = true;
            break;
        }

        std::string frame = sse_frame(tok);
        auto [write_ec, _] =
            co_await asio::async_write(socket, asio::buffer(frame), as_tuple(deferred));
        if (write_ec) {
            req->cancelled = true;
            break;
        }

        if (tok.finished) break;
    }
}

}  // namespace server
}  // namespace ccinfer
