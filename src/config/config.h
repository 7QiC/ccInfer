#pragma once

#include <string>

#include "cache/block.h"
#include "common/error_code.h"
#include "model/config.h"

namespace ccinfer {

struct EngineConfig {
    int device_id = 0;
    int max_blocks = 1024;
    int block_size = kKVBlockSize;
    int max_sequences = 64;
    int max_running_requests = 16;
    int max_concurrent_batches = 2;
    int max_pending_requests = 256;
    int max_token_budget = 4096;
    int max_seq_prefill_tokens = 512;
    int default_max_context_len = 2048;

    Result<void> validate() const;
};

struct Config {
    std::string model_path_;
    ModelConfig model_;
    EngineConfig engine_;

    static Result<Config> load(const std::string& model_path, EngineConfig engine = {});
};

}  // namespace ccinfer
