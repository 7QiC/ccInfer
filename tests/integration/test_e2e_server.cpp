#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>

class E2EServerTest : public ::testing::Test {
protected:
    pid_t server_pid_ = -1;
    int port_ = 0;
    std::string model_path_;

    static std::string g_server_binary;

    // --- socket helpers ---

    static void set_socket_timeout(int fd, int sec) {
        timeval tv{};
        tv.tv_sec = sec;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    static bool send_all(int fd, const char* data, size_t len) {
        size_t sent = 0;
        while (sent < len) {
            ssize_t n = send(fd, data + sent, len - sent, 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    int connect_socket() const {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return -1;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port_));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close(s);
            return -1;
        }
        set_socket_timeout(s, 10);
        return s;
    }

    static int find_free_port() {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return -1;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        addr.sin_port = 0;
        if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close(s);
            return -1;
        }
        sockaddr_in bound{};
        socklen_t len = sizeof(bound);
        if (getsockname(s, reinterpret_cast<sockaddr*>(&bound), &len) < 0) {
            close(s);
            return -1;
        }
        int p = ntohs(bound.sin_port);
        close(s);
        return p;
    }

    bool is_healthy() const {
        int s = connect_socket();
        if (s < 0) return false;
        const char* req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
        send_all(s, req, strlen(req));
        char resp[256]{};
        recv(s, resp, sizeof(resp) - 1, 0);
        close(s);
        return strstr(resp, "200 OK") != nullptr;
    }

    // --- SSE ---

    struct SSEStats {
        int tokens = 0;
        bool has_done = false;
    };

    static SSEStats parse_sse(const std::string& response) {
        SSEStats s;
        size_t pos = 0;
        while ((pos = response.find("data: ", pos)) != std::string::npos) {
            size_t end = response.find("\n\n", pos);
            if (end == std::string::npos) break;
            std::string frame = response.substr(pos, end - pos);
            s.has_done = s.has_done || (frame.find("\"done\":true") != std::string::npos);
            if (frame.find("\"token\"") != std::string::npos) ++s.tokens;
            pos = end + 2;
        }
        return s;
    }

    // --- HTTP ---

    static std::string escape_json(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '"' || c == '\\') out += '\\';
            out += c;
        }
        return out;
    }

    std::string http_request(const std::string& method, const std::string& path,
                             const std::string& body = {}) {
        int s = connect_socket();
        if (s < 0) return "";

        std::string req = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n";
        if (!body.empty()) {
            req += "Content-Type: application/json\r\nContent-Length: " +
                   std::to_string(body.size()) + "\r\n";
        }
        req += "Connection: close\r\n\r\n" + body;
        if (!send_all(s, req.c_str(), req.size())) {
            close(s);
            return "";
        }

        std::string response;
        char buf[4096];
        int n;
        while ((n = recv(s, buf, sizeof(buf) - 1, 0)) > 0) {
            response.append(buf, static_cast<size_t>(n));
        }
        close(s);
        return response;
    }

    std::string http_post(const std::string& path, const std::string& body) {
        return http_request("POST", path, body);
    }

    std::string http_get(const std::string& path) {
        return http_request("GET", path);
    }

    static std::string long_prompt(int approx_tokens) {
        std::ostringstream oss;
        for (int i = 0; i < approx_tokens; ++i) oss << " hello";
        return oss.str();
    }

    // --- test lifecycle ---

    void SetUp() override {
        model_path_ = std::getenv("CCINFER_TEST_MODEL_PATH")
                          ? std::getenv("CCINFER_TEST_MODEL_PATH")
                          : "../models/qwen3-0.6B";

        if (access((model_path_ + "/model.safetensors").c_str(), F_OK) != 0 ||
            access((model_path_ + "/config.json").c_str(), F_OK) != 0 ||
            access((model_path_ + "/tokenizer.json").c_str(), F_OK) != 0) {
            GTEST_SKIP() << "Model not found at " << model_path_;
        }

        port_ = find_free_port();
        ASSERT_GT(port_, 0) << "Could not find a free port";

        server_pid_ = fork();
        ASSERT_NE(server_pid_, -1) << "fork failed: " << strerror(errno);

        if (server_pid_ == 0) {
            execl(g_server_binary.c_str(), "ccinfer-server", "--port",
                  std::to_string(port_).c_str(), "--model-path", model_path_.c_str(), nullptr);
            _exit(127);
        }

        bool healthy = false;
        for (int i = 0; i < 120; ++i) {
            if (is_healthy()) {
                healthy = true;
                break;
            }
            if (waitpid(server_pid_, nullptr, WNOHANG) > 0) {
                FAIL() << "Server process exited prematurely (check model path: "
                       << model_path_ << ")";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        ASSERT_TRUE(healthy) << "Server did not become healthy within 60s";
    }

    void TearDown() override {
        if (server_pid_ <= 0) return;

        kill(server_pid_, SIGTERM);
        int status = 0;
        bool exited = false;
        for (int i = 0; i < 30; ++i) {
            pid_t r = waitpid(server_pid_, &status, WNOHANG);
            if (r == server_pid_) {
                exited = true;
                break;
            }
            if (r < 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (!exited) {
            kill(server_pid_, SIGKILL);
            waitpid(server_pid_, &status, 0);
        }
        if (exited) {
            EXPECT_TRUE(WIFEXITED(status)) << "Server did not exit cleanly";
            EXPECT_EQ(WEXITSTATUS(status), 0) << "Server exit code != 0";
        }
    }
};

#ifdef CCINFER_SERVER_BINARY
std::string E2EServerTest::g_server_binary = CCINFER_SERVER_BINARY;
#else
std::string E2EServerTest::g_server_binary = "./src/ccinfer-server";
#endif

// ---------------------------------------------------------------------------
// Health & routing
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, HealthEndpoint) {
    EXPECT_TRUE(is_healthy());
}

TEST_F(E2EServerTest, ModelsEndpoint) {
    std::string r = http_get("/v1/models");
    EXPECT_NE(r.find("200 OK"), std::string::npos);
}

TEST_F(E2EServerTest, NotFound) {
    std::string r = http_post("/nonexistent", "{}");
    EXPECT_NE(r.find("404 Not Found"), std::string::npos);
}

TEST_F(E2EServerTest, MethodNotAllowed) {
    // /v1/chat/completions requires POST
    std::string r = http_get("/v1/chat/completions");
    EXPECT_NE(r.find("405 Method Not Allowed"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Basic SSE
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, ChatCompletionsSSE) {
    std::string body =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0.0})";
    std::string r = http_post("/v1/chat/completions", body);

    EXPECT_NE(r.find("HTTP/1.1 200 OK"), std::string::npos);
    EXPECT_NE(r.find("text/event-stream"), std::string::npos);

    auto s = parse_sse(r);
    EXPECT_GE(s.tokens, 1);
    EXPECT_TRUE(s.has_done);
}

TEST_F(E2EServerTest, EmptyMessagesUsesFallback) {
    // Phase 4.1 fallback: empty messages → dummy tokens {1,2,3}.
    // Future: should return 400.
    std::string body =
        R"({"messages":[],"max_tokens":4,"temperature":0.0})";
    std::string r = http_post("/v1/chat/completions", body);
    EXPECT_NE(r.find("200 OK"), std::string::npos);
    EXPECT_TRUE(parse_sse(r).has_done);
}

TEST_F(E2EServerTest, ZeroMaxTokensReturnsDone) {
    std::string body =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":0,"temperature":0.0})";
    std::string r = http_post("/v1/chat/completions", body);
    EXPECT_NE(r.find("200 OK"), std::string::npos);
    EXPECT_TRUE(parse_sse(r).has_done);
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, BadJsonReturns400) {
    std::string r = http_post("/v1/chat/completions", "not json");
    EXPECT_NE(r.find("400 Bad Request"), std::string::npos);
}

TEST_F(E2EServerTest, MissingMaxTokensAccepted) {
    // max_tokens defaults to 256; should not crash.
    std::string body = R"({"messages":[{"role":"user","content":"Hi"}]})";
    std::string r = http_post("/v1/chat/completions", body);
    EXPECT_NE(r.find("200 OK"), std::string::npos);
}

TEST_F(E2EServerTest, InvalidTemperatureReturns400) {
    std::string body =
        R"({"messages":[{"role":"user","content":"Hi"}],"temperature":"cold"})";
    std::string r = http_post("/v1/chat/completions", body);
    EXPECT_NE(r.find("400 Bad Request"), std::string::npos);
}

TEST_F(E2EServerTest, NegativeMaxTokensReturns400) {
    std::string body =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":-1})";
    std::string r = http_post("/v1/chat/completions", body);
    EXPECT_NE(r.find("400 Bad Request"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Chunked prefill
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, LongPromptChunkedPrefill) {
    std::string prompt = long_prompt(800);
    std::ostringstream body;
    body << R"({"messages":[{"role":"user","content":")" << escape_json(prompt)
         << R"("}],"max_tokens":8,"temperature":0.0})";
    std::string r = http_post("/v1/chat/completions", body.str());

    EXPECT_NE(r.find("200 OK"), std::string::npos);
    auto s = parse_sse(r);
    EXPECT_GE(s.tokens, 1);
    EXPECT_TRUE(s.has_done);
}

// ---------------------------------------------------------------------------
// Decode continuation
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, DecodeContinuationManyTokens) {
    std::string body =
        R"({"messages":[{"role":"user","content":"Once upon a time"}],"max_tokens":32,"temperature":0.0})";
    std::string r = http_post("/v1/chat/completions", body);

    EXPECT_NE(r.find("200 OK"), std::string::npos);
    auto s = parse_sse(r);
    EXPECT_GE(s.tokens, 8);
    EXPECT_TRUE(s.has_done);
}

// ---------------------------------------------------------------------------
// Concurrent requests
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, ConcurrentShortRequests) {
    std::string body =
        R"({"messages":[{"role":"user","content":"A"}],"max_tokens":4,"temperature":0.0})";

    std::string r1, r2;
    std::thread t1([&] { r1 = http_post("/v1/chat/completions", body); });
    std::thread t2([&] { r2 = http_post("/v1/chat/completions", body); });
    t1.join();
    t2.join();

    EXPECT_TRUE(parse_sse(r1).has_done);
    EXPECT_TRUE(parse_sse(r2).has_done);
}

TEST_F(E2EServerTest, ConcurrentLongAndShort) {
    std::string prompt = long_prompt(600);
    std::ostringstream long_body;
    long_body << R"({"messages":[{"role":"user","content":")" << escape_json(prompt)
              << R"("}],"max_tokens":8,"temperature":0.0})";

    std::string short_body =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0.0})";

    std::string r1, r2;
    std::thread t1([&] { r1 = http_post("/v1/chat/completions", long_body.str()); });
    std::thread t2([&] { r2 = http_post("/v1/chat/completions", short_body); });
    t1.join();
    t2.join();

    EXPECT_TRUE(parse_sse(r1).has_done) << "Long prompt request should complete";
    EXPECT_TRUE(parse_sse(r2).has_done) << "Short prompt request should complete";
}
