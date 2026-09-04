#pragma once

#include <vector>

#include "backend/backend.h"
#include "base/error.h"
#include "config/model_config.h"
#include "model/weight_source.h"
#include "runtime/tensor.h"

namespace ccinfer {

struct Qwen35GdnLayerWeights {
    Tensor attn_norm;
    Tensor attn_qkv;
    Tensor attn_gate;
    Tensor ssm_conv1d;
    Tensor ssm_alpha;
    Tensor ssm_beta;
    Tensor ssm_out;
    Tensor ssm_norm;
    Tensor ssm_a;
    Tensor ssm_dt_bias;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
    Tensor post_attention_norm;
};

struct Qwen35AttnLayerWeights {
    Tensor attn_norm;
    Tensor attn_q;
    Tensor attn_gate;
    Tensor attn_k;
    Tensor attn_v;
    Tensor attn_q_norm;
    Tensor attn_k_norm;
    Tensor attn_output;
    Tensor ffn_gate;
    Tensor ffn_up;
    Tensor ffn_down;
    Tensor post_attention_norm;
};

struct Qwen35Weights {
    Tensor embed;
    Tensor lm_head;  // Tied to embed: shares the same Buffer/Tensor.
    Tensor rms_final;
    std::vector<Qwen35GdnLayerWeights> gdn_layers_;
    std::vector<Qwen35AttnLayerWeights> attn_layers_;

    static Result<Qwen35Weights> load(Backend& backend, const ModelConfig& config,
                                      WeightSource& weights);
};

}  // namespace ccinfer
