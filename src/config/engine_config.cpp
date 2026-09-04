#include "config/engine_config.h"

#include <cstdint>
#include <limits>

namespace ccinfer {

Result<void> EngineConfig::validate() const {
    if (device_id < 0 || max_blocks <= 0 || kv_block_size <= 0 || max_sequences <= 0 ||
        max_running_requests <= 0 || max_running_requests > max_sequences ||
        max_concurrent_batches <= 0 || max_pending_requests <= 0 || max_token_budget <= 0 ||
        max_seq_prefill_tokens < 0 || default_max_context_len <= 0 ||
        state_prefix_cache_blocks < 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const auto max_context_capacity = static_cast<int64_t>(max_blocks) * kv_block_size;
    if (max_context_capacity <= 0 ||
        max_context_capacity > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        default_max_context_len > max_context_capacity) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return {};
}

}  // namespace ccinfer
