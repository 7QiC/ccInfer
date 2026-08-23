#include "config/config.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <utility>

#include <nlohmann/json.hpp>

namespace ccinfer {

Result<void> EngineConfig::validate() const {
    if (device_id < 0 || max_blocks <= 0 || block_size <= 0 || max_sequences <= 0 ||
        max_active_scheduled_sequences <= 0 || schedule_compute_budget <= 0 ||
        prefill_chunk_size <= 0 || default_max_context_len <= 0 || dummy_num_layers <= 0 ||
        dummy_num_kv_heads <= 0 || dummy_head_dim <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const auto max_context_capacity = static_cast<int64_t>(max_blocks) * block_size;
    if (max_context_capacity <= 0 ||
        max_context_capacity > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        default_max_context_len > max_context_capacity) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return {};
}

Result<Config> Config::load(const std::string& model_path, EngineConfig engine) {
    if (model_path.empty()) return std::unexpected(ErrorCode::ModelLoadFailed);
    auto engine_r = engine.validate();
    if (!engine_r) return std::unexpected(engine_r.error());

    std::ifstream cfg_file(model_path + "/config.json");
    if (!cfg_file.is_open()) return std::unexpected(ErrorCode::ModelLoadFailed);

    auto json = nlohmann::json::parse(cfg_file, nullptr, false);
    if (json.is_discarded()) return std::unexpected(ErrorCode::ModelConfigInvalid);

    auto model_r = ModelConfig::from_json(json);
    if (!model_r) return std::unexpected(model_r.error());

    return Config{.model_path_ = model_path, .model_ = std::move(*model_r), .engine_ = engine};
}

}  // namespace ccinfer
