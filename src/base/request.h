#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <boost/asio/any_io_executor.hpp>

#include "base/channel.h"
#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

// HTTP -> Scheduler request boundary.
// TokenSink lets Scheduler post tokens/events back to the HTTP io_context.
struct TokenSink {
    asio::any_io_executor executor;
    std::weak_ptr<TokenChannel> channel;
    std::function<void()> on_send_failed;
};

struct SchedulerRequest {
    std::string request_id;
    std::vector<int32_t> prompt_tokens;
    SamplingParams sampling;
    int max_context_len = 0;  // 0 means use EngineConfig::default_max_context_len.
    TokenSink sink;
};

}  // namespace ccinfer
