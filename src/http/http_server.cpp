#include "http/http_server.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/read_until.hpp>
#include <boost/asio/this_coro.hpp>
#include <boost/asio/write.hpp>
#include <nlohmann/json.hpp>

#include "common/types.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"

namespace ccinfer {

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

constexpr std::string_view kInternalError =
    "HTTP/1.1 500 Internal Server Error\r\n"
    "Content-Length: 0\r\n"
    "Connection: close\r\n"
    "\r\n";

std::string sse_error_frame(ErrorCode err) {
    nlohmann::json j;
    j["error"] = std::string(error_message(err));
    j["done"] = true;
    return "data: " + j.dump() + "\n\n";
}

std::string json_response(const nlohmann::json& body) {
    std::string payload = body.dump();
    return "HTTP/1.1 200 OK\r\n"
           "Content-Type: application/json\r\n"
           "Content-Length: " +
           std::to_string(payload.size()) +
           "\r\n"
           "Connection: close\r\n"
           "\r\n" +
           payload;
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

constexpr size_t kMaxBodySize = 64 * 1024 * 1024;  // 64 MB

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
        if (out > kMaxBodySize / 10) return {CLState::Invalid, 0};
        out = out * 10 + static_cast<size_t>(ch - '0');
        if (out > kMaxBodySize) return {CLState::Invalid, 0};
    }
    if (!has_digit) return {CLState::Invalid, 0};
    for (; i < val.size(); ++i) {
        if (!std::isspace(static_cast<unsigned char>(val[i]))) return {CLState::Invalid, 0};
    }
    return {CLState::Valid, out};
}

}  // namespace

HttpServer::HttpServer(asio::io_context& io, uint16_t port, Scheduler& scheduler,
                       Tokenizer& tokenizer)
    : io_(io), port_(port), scheduler_(scheduler), tokenizer_(tokenizer), acceptor_(io) {}

// Owner must call shutdown() + shutdown_async().wait() before destroying.
HttpServer::~HttpServer() = default;

Result<void> HttpServer::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return {};

    boost::system::error_code ec;

    acceptor_.open(asio::ip::tcp::v4(), ec);
    if (ec) {
        running_ = false;
        return std::unexpected(ErrorCode::NetworkBindFailed);
    }

    acceptor_.set_option(asio::ip::tcp::acceptor::reuse_address(true), ec);
    if (ec) {
        running_ = false;
        return std::unexpected(ErrorCode::NetworkBindFailed);
    }

    acceptor_.bind(asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port_), ec);
    if (ec) {
        running_ = false;
        return std::unexpected(ErrorCode::NetworkBindFailed);
    }

    acceptor_.listen(asio::socket_base::max_listen_connections, ec);
    if (ec) {
        running_ = false;
        return std::unexpected(ErrorCode::NetworkBindFailed);
    }

    asio::co_spawn(io_, accept_loop(), detached);
    return {};
}

void HttpServer::shutdown() { shutdown_async(); }

std::shared_future<void> HttpServer::shutdown_async() {
    std::lock_guard lock(shutdown_mutex_);

    if (shutdown_promise_) {
        return shutdown_future_;
    }

    shutdown_promise_ = std::make_unique<std::promise<void>>();
    shutdown_future_ = shutdown_promise_->get_future().share();

    bool was_running = running_.exchange(false);

    asio::post(io_, [this, was_running] {
        boost::system::error_code ec;
        acceptor_.cancel(ec);
        acceptor_.close(ec);

        // Cancel all active connections: close socket + close channel + cancel
        // scheduler request.  Closing the channel unblocks async_receive so
        // the SSE coroutine can exit.
        for (auto& conn : active_conns_) {
            conn->socket->cancel(ec);
            conn->socket->close(ec);
            if (conn->channel) conn->channel->close();
            if (!conn->request_id.empty()) scheduler_.cancel(conn->request_id);
        }

        if (!was_running) {
            accept_loop_done_ = true;
        }

        try_finish_shutdown_on_http_thread();
    });

    return shutdown_future_;
}

void HttpServer::try_finish_shutdown_on_http_thread() {
    if (shutdown_done_sent_) return;
    if (!running_ && accept_loop_done_ && active_conns_.empty()) {
        std::lock_guard lock(shutdown_mutex_);
        if (shutdown_promise_) {
            shutdown_done_sent_ = true;
            shutdown_promise_->set_value();
        }
    }
}

asio::awaitable<void> HttpServer::accept_loop() {
    co_await accept_loop_impl();
    accept_loop_done_ = true;
    try_finish_shutdown_on_http_thread();
}

asio::awaitable<void> HttpServer::accept_loop_impl() {
    while (running_) {
        auto [acc_ec, socket] = co_await acceptor_.async_accept(as_tuple(deferred));
        if (acc_ec) {
            if (!running_) break;
            continue;
        }
        auto sock = std::make_shared<asio::ip::tcp::socket>(std::move(socket));
        auto conn = std::make_shared<ActiveConn>();
        conn->socket = sock;
        active_conns_.insert(conn);
        asio::co_spawn(io_, handle_connection(std::move(sock), conn), detached);
    }
}

asio::awaitable<std::string> HttpServer::read_line(asio::ip::tcp::socket& socket,
                                                   asio::streambuf& buf) {
    std::string line;
    while (true) {
        if (buf.size() > 0) {
            std::vector<char> avail(buf.size());
            const std::size_t avail_n =
                buf.sgetn(avail.data(), static_cast<std::streamsize>(avail.size()));
            auto it = std::find(avail.begin(), avail.begin() + static_cast<std::ptrdiff_t>(avail_n),
                                '\n');
            if (it != avail.begin() + static_cast<std::ptrdiff_t>(avail_n)) {
                std::size_t len = static_cast<std::size_t>(it - avail.begin());
                if (len > 0 && avail[len - 1] == '\r') --len;
                if (line.size() + len > kMaxHeaderSize) {
                    co_return std::string(kMaxHeaderSize + 1, 'x');
                }
                line.append(avail.data(), len);
                const std::size_t consumed = static_cast<std::size_t>(it - avail.begin()) + 1;
                if (consumed < avail_n) {
                    buf.sputn(avail.data() + static_cast<std::ptrdiff_t>(consumed),
                              static_cast<std::streamsize>(avail_n - consumed));
                }
                co_return line;
            }
            if (line.size() + avail_n > kMaxHeaderSize) {
                co_return std::string(kMaxHeaderSize + 1, 'x');
            }
            line.append(avail.data(), avail_n);
        }

        char chunk[4096];
        auto [ec, n] = co_await socket.async_read_some(asio::buffer(chunk), as_tuple(deferred));
        if (ec || n == 0) co_return "";
        buf.sputn(chunk, static_cast<std::streamsize>(n));
    }
}

asio::awaitable<std::optional<std::string>> HttpServer::read_body(asio::ip::tcp::socket& socket,
                                                                  asio::streambuf& buf,
                                                                  size_t content_length) {
    if (content_length > kMaxBodySize) co_return std::nullopt;

    std::string body;
    body.reserve(content_length);

    auto avail = buf.size();
    if (avail > 0) {
        size_t take = std::min(avail, content_length);
        body.resize(take);
        buf.sgetn(body.data(), static_cast<std::streamsize>(take));
    }

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

asio::awaitable<void> HttpServer::handle_connection(std::shared_ptr<asio::ip::tcp::socket> socket,
                                                    std::shared_ptr<ActiveConn> conn) {
    co_await handle_connection_impl(*socket, conn);
    active_conns_.erase(conn);
    try_finish_shutdown_on_http_thread();
}

asio::awaitable<void> HttpServer::handle_connection_impl(asio::ip::tcp::socket& socket,
                                                         std::shared_ptr<ActiveConn> conn) {
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

    auto qpos = path.find('?');
    if (qpos != std::string::npos) path.resize(qpos);

    // Read headers with size limit (start counting from request line).
    size_t content_length = 0;
    bool has_content_length = false;
    bool content_length_malformed = false;
    size_t header_bytes = request_line.size() + 2;  // +2 for \r\n
    while (true) {
        auto line = co_await read_line(socket, buf);
        if (line.empty()) break;
        header_bytes += line.size() + 2;  // +2 for \r\n
        if (header_bytes > kMaxHeaderSize) {
            co_await write_response(socket, kBadRequest);
            co_return;
        }
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
    } else if (path == "/metrics") {
        if (method != "GET") {
            co_await write_response(socket, kMethodNotAllowed);
            co_return;
        }
        co_await handle_metrics(socket);
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
        co_await handle_chat(socket, std::move(*body), conn);
    } else {
        co_await write_response(socket, kNotFoundResponse);
    }
}

std::string HttpServer::make_sse_frame(const GeneratedToken& tok) {
    nlohmann::json j;
    if (tok.has_token) {
        auto decoded = tokenizer_.decode({tok.token_id}, false);
        j["token"] = decoded ? *decoded : "";
        j["token_id"] = tok.token_id;
    }
    j["done"] = tok.finished;
    return "data: " + j.dump() + "\n\n";
}

asio::awaitable<void> HttpServer::handle_health(asio::ip::tcp::socket& socket) {
    co_await write_response(socket, kHealthResponse);
}

asio::awaitable<void> HttpServer::handle_models(asio::ip::tcp::socket& socket) {
    co_await write_response(socket, kModelsResponse);
}

asio::awaitable<void> HttpServer::handle_metrics(asio::ip::tcp::socket& socket) {
    auto cap = scheduler_.capacity();
    nlohmann::json j;
    j["engine"] = {
        {"max_sequences", cap.max_sequences},
        {"active_sequences", cap.active_sequences},
    };

    double pool_util = 0.0;
    if (cap.max_blocks > 0) {
        pool_util = static_cast<double>(cap.block_active + cap.block_cached_idle) /
                    static_cast<double>(cap.max_blocks);
    }
    j["kv_cache"] = {
        {"block_total", cap.max_blocks},
        {"block_free", cap.free_blocks},
        {"block_active", cap.block_active},
        {"block_cached_idle", cap.block_cached_idle},
        {"block_size", cap.block_size},
        {"pool_utilization", pool_util},
        {"active_capacity_tokens", cap.block_active * cap.block_size},
        {"cached_capacity_tokens", cap.block_cached_idle * cap.block_size},
    };
    j["prefix_cache"] = {
        {"lookup_hits", cap.prefix_lookup_hits},
        {"lookup_misses", cap.prefix_lookup_misses},
        {"evictions", cap.prefix_evictions},
        {"cached_blocks", cap.prefix_cached_blocks},
    };

    co_await write_response(socket, json_response(j));
}

namespace {

enum class JsonFieldState { Missing, Valid, Invalid };

template <typename T>
JsonFieldState safe_json_get(const nlohmann::json& j, const char* key, T& out) {
    if (!j.contains(key)) return JsonFieldState::Missing;
    const auto& val = j[key];
    if constexpr (std::is_same_v<T, int>) {
        if (!val.is_number_integer()) return JsonFieldState::Invalid;
        auto raw = val.get<int64_t>();
        if (raw < std::numeric_limits<int>::min() || raw > std::numeric_limits<int>::max())
            return JsonFieldState::Invalid;
        out = static_cast<int>(raw);
        return JsonFieldState::Valid;
    } else if constexpr (std::is_same_v<T, float>) {
        if (!val.is_number()) return JsonFieldState::Invalid;
        float v = val.get<float>();
        if (!std::isfinite(v)) return JsonFieldState::Invalid;
        out = v;
        return JsonFieldState::Valid;
    }
    return JsonFieldState::Invalid;
}

struct ParsedChatRequest {
    std::vector<int32_t> prompt_tokens;
    SamplingParams sampling;
    int max_context_len = 0;
};

Result<ParsedChatRequest> parse_chat_request(const std::string& body, Tokenizer& tokenizer) {
    nlohmann::json req_json = nlohmann::json::parse(body.empty() ? "{}" : body, nullptr, false);
    if (req_json.is_discarded() || !req_json.is_object()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    ParsedChatRequest parsed;
    if (req_json.contains("messages")) {
        if (!req_json["messages"].is_array()) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        for (const auto& msg : req_json["messages"]) {
            if (!msg.is_object() || !msg.contains("content") || !msg["content"].is_string()) {
                return std::unexpected(ErrorCode::InvalidArgument);
            }
            auto encoded = tokenizer.encode(msg["content"].get<std::string>());
            if (!encoded) return std::unexpected(ErrorCode::InternalError);
            parsed.prompt_tokens.insert(parsed.prompt_tokens.end(), encoded->begin(),
                                        encoded->end());
        }
    }
    // Keep the minimal control-path prompt for the dummy worker and empty-message API case.
    if (parsed.prompt_tokens.empty()) parsed.prompt_tokens = {1, 2, 3};

    auto read_optional = [&](const char* key, auto& target) -> bool {
        return safe_json_get(req_json, key, target) != JsonFieldState::Invalid;
    };
    if (!read_optional("max_tokens", parsed.sampling.max_tokens) ||
        !read_optional("temperature", parsed.sampling.temperature) ||
        !read_optional("top_p", parsed.sampling.top_p) ||
        !read_optional("top_k", parsed.sampling.top_k)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (parsed.sampling.max_tokens < 0 || parsed.sampling.temperature < 0.0f ||
        parsed.sampling.top_p <= 0.0f || parsed.sampling.top_p > 1.0f ||
        parsed.sampling.top_k < 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto max_context_state = safe_json_get(req_json, "max_context_len", parsed.max_context_len);
    if (max_context_state == JsonFieldState::Invalid ||
        (max_context_state == JsonFieldState::Valid && parsed.max_context_len <= 0)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    return parsed;
}

}  // namespace

asio::awaitable<void> HttpServer::handle_chat(asio::ip::tcp::socket& socket, std::string body,
                                              const std::shared_ptr<ActiveConn>& conn) {
    auto parsed = parse_chat_request(body, tokenizer_);
    if (!parsed) {
        co_await write_response(
            socket, parsed.error() == ErrorCode::InternalError ? kInternalError : kBadRequest);
        co_return;
    }

    SchedulerRequest sreq;
    sreq.max_context_len = parsed->max_context_len;

    // Build HTTP-side state — request_id and channel owned by this coroutine.
    auto executor = co_await asio::this_coro::executor;
    auto request_id = "req-" + std::to_string(next_request_id_++);
    auto channel = std::make_shared<TokenChannel>(executor, 16);

    // Store in ActiveConn so shutdown() can close socket and channel.
    if (conn) {
        conn->request_id = request_id;
        conn->channel = channel;
    }

    sreq.request_id = request_id;
    sreq.prompt_tokens = std::move(parsed->prompt_tokens);
    sreq.sampling = parsed->sampling;
    sreq.sink.executor = executor;
    sreq.sink.channel = channel;
    sreq.sink.on_send_failed = [&sched = scheduler_, request_id]() { sched.cancel(request_id); };

    scheduler_.submit(std::move(sreq));

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
            scheduler_.cancel(request_id);
            co_return;
        }
    }

    while (true) {
        auto [recv_ec, result] = co_await channel->async_receive(as_tuple(deferred));
        if (recv_ec) {
            scheduler_.cancel(request_id);
            break;
        }

        std::string frame;
        bool done = false;
        if (result) {
            frame = make_sse_frame(*result);
            done = result->finished;
        } else {
            frame = sse_error_frame(result.error());
            done = true;
        }

        auto [write_ec, _] =
            co_await asio::async_write(socket, asio::buffer(frame), as_tuple(deferred));
        if (write_ec) {
            scheduler_.cancel(request_id);
            break;
        }

        if (done) break;
    }
}

}  // namespace ccinfer
