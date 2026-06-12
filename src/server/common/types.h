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

// TokenSink depends on TokenChannel (from common/channel.h) and
// GeneratedToken (from common/types.h). Both includes must precede this point.
//
// Scheduler posts tokens/events to HTTP io_context through this sink.
// The on_send_failed callback is invoked on the HTTP executor when the
// channel is gone or try_send fails.  Callers must check for null before
// invoking (default-constructed std::function evaluates to false).
//
// IMPORTANT: on_send_failed must NOT directly mutate scheduler state.
// It should post a cancel/notification back to the scheduler thread.
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

// Scheduler-owned request state — only accessed on scheduler_io thread.
struct SchedulerRequestState {
    std::string request_id;
    SequenceId seq_id = 0;

    std::vector<int32_t> initial_prompt_tokens;
    std::vector<int32_t> prompt_tokens;
    int prefill_cursor = 0;  // number of prompt tokens already consumed / committed
    bool prefill_done = false;
    bool suspended = false;
    int generated_in_prompt = 0;

    int32_t last_token = -1;   // next decode input; sampled but not yet in KV cache
    int tokens_generated = 0;  // output tokens already sent to client
    std::vector<int32_t> generated_tokens;
    bool finished = false;
    bool cancelled = false;

    SamplingParams sampling;
    int max_context_len = 2048;

    TokenSink sink;
};

}  // namespace server
}  // namespace ccinfer
