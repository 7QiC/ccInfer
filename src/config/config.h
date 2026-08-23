#pragma once

#include <string>

#include "base/result.h"
#include "cache/block.h"
#include "model/config.h"

namespace ccinfer {

struct EngineConfig {
    int device_id = 0;
    int max_blocks = 1024;
    int block_size = kKVBlockSize;
    int max_sequences = 64;
    int max_active_scheduled_sequences = 16;
    int schedule_compute_budget = 4096;
    int prefill_chunk_size = 512;
    int default_max_context_len = 2048;

    // Used only when the Worker is intentionally initialized without a model.
    int dummy_num_layers = 32;
    int dummy_num_kv_heads = 8;
    int dummy_head_dim = 128;

    Result<void> validate() const;
};

struct Config {
    std::string model_path_;
    ModelConfig model_;
    EngineConfig engine_;

    static Result<Config> load(const std::string& model_path, EngineConfig engine = {});
};

}  // namespace ccinfer
