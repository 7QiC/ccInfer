#include "config/model_config.h"

#include <cmath>

namespace ccinfer {

Result<void> ModelConfig::validate() const {
    if (arch_ == ModelArch::Unknown) {
        return std::unexpected(ErrorCode::ModelUnsupportedArch);
    }
    if (n_layers_ <= 0 || n_q_heads_ <= 0 || n_kv_heads_ <= 0 || head_dim_ <= 0 || d_model_ <= 0 ||
        d_ff_ <= 0 || vocab_size_ <= 0 || max_seq_len_ <= 0) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }
    if (n_kv_heads_ > n_q_heads_ || n_q_heads_ % n_kv_heads_ != 0) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }
    if (!std::isfinite(rope_theta_) || rope_theta_ <= 0.0f || !std::isfinite(rms_norm_eps_) ||
        rms_norm_eps_ <= 0.0f) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }

    if (arch_ == ModelArch::Qwen3_5) {
        if (full_attention_interval_ <= 0 || ssm_conv_kernel_ <= 0 || ssm_state_size_ <= 0 ||
            ssm_group_count_ <= 0 || ssm_time_step_rank_ <= 0 || ssm_inner_size_ <= 0 ||
            rotary_dim_ <= 0 || rotary_dim_ > head_dim_ || rotary_dim_ % 2 != 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        if (ssm_inner_size_ % ssm_time_step_rank_ != 0 ||
            ssm_inner_size_ / ssm_time_step_rank_ != ssm_state_size_) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }

        const int expected_layer_count = n_layers_ + nextn_predict_layers_;
        if (static_cast<int>(layer_types_.size()) != expected_layer_count) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        for (LayerType type : layer_types_) {
            if (type == LayerType::FullAttention || type == LayerType::GatedDeltaNet ||
                type == LayerType::MtpPredictor) {
                continue;
            }
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
    }
    return {};
}

}  // namespace ccinfer
