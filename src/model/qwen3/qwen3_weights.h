#pragma once

#include <memory>
#include <vector>

#include <cuda_bf16.h>

#include "backend/backend.h"
#include "common/error_code.h"
#include "core/tensor.h"
#include "model/config.h"
#include "model/loader.h"

namespace ccinfer {

struct Qwen3LayerWeights {
    Tensor qkv;
    Tensor o;
    Tensor gate;
    Tensor up;
    Tensor down;
    Tensor rms_attn;
    Tensor rms_ffn;
    Tensor q_norm;
    Tensor k_norm;
};

struct Qwen3Weights {
    Tensor embed;
    Tensor lm_head;
    Tensor rms_final;
    std::vector<Qwen3LayerWeights> layers_;

    static Result<Qwen3Weights> load(Backend& backend, const ModelConfig& config,
                                     const WeightLoader& loader);
};

}  // namespace ccinfer
