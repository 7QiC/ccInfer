#include "config/model_config.h"

#include <cmath>

namespace ccinfer {

Result<void> ModelConfig::validate() const {
    if (arch_ == ModelArch::Unknown) {
        return std::unexpected(ErrorCode::ModelUnsupportedArch);
    }
    if (n_layers_ <= 0 || n_q_heads_ <= 0 || n_kv_heads_ <= 0 || head_dim_ <= 0 ||
        d_model_ <= 0 || d_ff_ <= 0 || vocab_size_ <= 0 || max_seq_len_ <= 0) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }
    if (n_kv_heads_ > n_q_heads_ || n_q_heads_ % n_kv_heads_ != 0) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }
    if (!std::isfinite(rope_theta_) || rope_theta_ <= 0.0f || !std::isfinite(rms_norm_eps_) ||
        rms_norm_eps_ <= 0.0f) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }
    return {};
}

}  // namespace ccinfer
