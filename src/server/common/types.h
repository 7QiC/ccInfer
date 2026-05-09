#pragma once

#include <boost/asio/any_io_executor.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "common/channel.h"
#include "common/types.h"

namespace ccinfer {
namespace server {

namespace asio = boost::asio;

struct SamplingParams {
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
};

// TokenSink depends on TokenChannel (from common/channel.h) and
// GeneratedToken (from common/types.h). Both includes must precede this point.
//
// Scheduler posts tokens/events to HTTP io_context through this sink.
// The on_send_failed callback is invoked on the HTTP executor when the
// channel is gone or try_send fails; it should post a cancel back to
// the scheduler.
struct TokenSink {
    asio::any_io_executor executor;
    std::weak_ptr<TokenChannel> channel;
    std::function<void()> on_send_failed;
};

// Lightweight request passed from HTTP thread to Scheduler thread.
struct SchedulerRequest {
    std::string request_id;
    std::vector<int32_t> prompt_tokens;
    SamplingParams sampling;
    int max_context_len = 2048;
    TokenSink sink;
};

// HTTP-owned connection state — only accessed on http_io thread.
struct ConnectionState {
    std::string request_id;
    std::shared_ptr<TokenChannel> output_channel;
    bool closed = false;
};

// Scheduler-owned request state — only accessed on scheduler_io thread.
struct SchedulerRequestState {
    std::string request_id;
    SequenceId seq_id = 0;

    std::vector<int32_t> prompt_tokens;
    int prefill_cursor = 0;
    bool prefill_done = false;

    int32_t last_token = -1;
    int tokens_generated = 0;
    bool finished = false;
    bool cancelled = false;

    SamplingParams sampling;
    int max_context_len = 2048;

    TokenSink sink;
};

}  // namespace server
}  // namespace ccinfer
