#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/channel.h"
#include "common/types.h"

namespace ccinfer {
namespace server {

struct SamplingParams {
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 0.9f;
    int top_k = 50;
};

struct RequestState {
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

    std::shared_ptr<TokenChannel> output_channel;
};

}  // namespace server
}  // namespace ccinfer
