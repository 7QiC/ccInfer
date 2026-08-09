#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <vector>

#include "base/result.h"
#include "backend/backend.h"
#include "model/config.h"
#include "model/loader.h"

namespace ccinfer {

struct Qwen3LayerWeights {
    std::shared_ptr<Buffer> qkv_;
    std::shared_ptr<Buffer> o_;
    std::shared_ptr<Buffer> gate_;
    std::shared_ptr<Buffer> up_;
    std::shared_ptr<Buffer> down_;
    std::shared_ptr<Buffer> rms_attn_;
    std::shared_ptr<Buffer> rms_ffn_;
    std::shared_ptr<Buffer> q_norm_;
    std::shared_ptr<Buffer> k_norm_;
};

struct Qwen3Weights {
    std::shared_ptr<Buffer> embed_;
    std::shared_ptr<Buffer> lm_head_;
    std::shared_ptr<Buffer> rms_final_;
    std::vector<Qwen3LayerWeights> layers_;

    static Result<Qwen3Weights> load(Backend& backend, const ModelConfig& config,
                                     const WeightLoader& loader);
};

}  // namespace ccinfer
