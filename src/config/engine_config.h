#pragma once

#include <string>

#include "base/error.h"

namespace ccinfer {

// ccInfer runtime resource/policy configuration. It contains no model
// architecture fields.
struct EngineConfig {
    int device_id = 0;
    int max_blocks = 1024;
    int kv_block_size = 128;
    int max_sequences = 64;
    int max_running_requests = 16;
    int max_concurrent_batches = 2;
    int max_pending_requests = 256;
    int max_token_budget = 4096;
    int max_seq_prefill_tokens = 512;
    int default_max_context_len = 2048;

    // Number of leading full-KV-block frontiers that may hold immutable GDN
    // state snapshots. 0 disables state prefix caching; 3 is the M1 default.
    int state_prefix_cache_blocks = 3;

    Result<void> validate() const;
};

// Startup-only aggregate for main/http wiring. It intentionally does not own a
// ModelConfig; ModelConfig is produced by Checkpoint/HF/GGUF loaders.
struct LaunchConfig {
    std::string model_path;
    EngineConfig engine;
};

}  // namespace ccinfer
