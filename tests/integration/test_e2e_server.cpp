#include <arpa/inet.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

class E2EServerTest : public ::testing::Test {
protected:
    pid_t server_pid_ = -1;
    int port_ = 0;
    std::string model_path_;

    static std::string g_server_binary;

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
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return false;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port_));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close(s);
            return false;
        }
        const char* req = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
        send(s, req, strlen(req), 0);
        char resp[256]{};
        recv(s, resp, sizeof(resp) - 1, 0);
        close(s);
        return strstr(resp, "200 OK") != nullptr;
    }

    // Returns HTTP response body.
    std::string http_post(const std::string& path, const std::string& body) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        if (s < 0) return "";
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port_));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            close(s);
            return "";
        }

        std::string req = "POST " + path +
                          " HTTP/1.1\r\n"
                          "Host: 127.0.0.1\r\n"
                          "Content-Type: application/json\r\n"
                          "Content-Length: " +
                          std::to_string(body.size()) +
                          "\r\n"
                          "Connection: close\r\n"
                          "\r\n" +
                          body;
        send(s, req.c_str(), req.size(), 0);

        std::string response;
        char buf[4096];
        int n;
        while ((n = recv(s, buf, sizeof(buf) - 1, 0)) > 0) {
            response.append(buf, static_cast<size_t>(n));
        }
        close(s);
        return response;
    }

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
            if (waitpid(server_pid_, nullptr, WNOHANG) != 0) {
                FAIL() << "Server process exited prematurely (check model path: " << model_path_
                       << ")";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        ASSERT_TRUE(healthy) << "Server did not become healthy within 60s";
    }

    void TearDown() override {
        if (server_pid_ > 0) {
            kill(server_pid_, SIGTERM);
            int status = 0;
            int waited = 0;
            while (waitpid(server_pid_, &status, WNOHANG) == 0 && waited < 30) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                ++waited;
            }
            if (waitpid(server_pid_, &status, WNOHANG) != 0) {
                kill(server_pid_, SIGKILL);
                waitpid(server_pid_, &status, 0);
            }
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
// Tests
// ---------------------------------------------------------------------------

TEST_F(E2EServerTest, HealthEndpoint) {
    // Already verified in SetUp; test idempotency.
    EXPECT_TRUE(is_healthy());
}

TEST_F(E2EServerTest, ChatCompletionsSSE) {
    std::string body =
        R"({"messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0.0})";
    std::string response = http_post("/v1/chat/completions", body);

    // Should have SSE headers
    EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
    EXPECT_NE(response.find("text/event-stream"), std::string::npos);

    // Parse SSE frames: "data: ...\n\n"
    int token_count = 0;
    bool has_done = false;
    size_t pos = 0;
    while ((pos = response.find("data: ", pos)) != std::string::npos) {
        size_t end = response.find("\n\n", pos);
        if (end == std::string::npos) break;
        std::string frame = response.substr(pos, end - pos);
        has_done = has_done || (frame.find("\"done\":true") != std::string::npos);
        if (frame.find("\"token\"") != std::string::npos) ++token_count;
        pos = end + 2;
    }

    EXPECT_GE(token_count, 1) << "Expected at least 1 token in SSE stream";
    EXPECT_TRUE(has_done) << "Final SSE frame should contain \"done\":true";
}

TEST_F(E2EServerTest, ChatCompletionsEmptyMessages) {
    std::string body =
        R"({"messages":[],"max_tokens":4,"temperature":0.0})";
    std::string response = http_post("/v1/chat/completions", body);

    // Empty messages triggers Phase 4.1 fallback (dummy tokens {1,2,3}).
    EXPECT_NE(response.find("HTTP/1.1 200 OK"), std::string::npos);
    bool has_done = (response.find("\"done\":true") != std::string::npos);
    EXPECT_TRUE(has_done);
}

TEST_F(E2EServerTest, NotFound) {
    std::string response = http_post("/nonexistent", "{}");
    EXPECT_NE(response.find("404 Not Found"), std::string::npos);
}
