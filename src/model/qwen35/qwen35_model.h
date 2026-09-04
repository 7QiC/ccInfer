#pragma once

#include "config/model_config.h"

namespace ccinfer {
namespace qwen35 {

// Qwen3.5-specific hybrid layer mapping. These helpers deliberately live with
// the Qwen3.5 model code rather than in the generic ModelConfig.
int num_kv_layers(const ModelConfig& config);
int kv_layer_index(const ModelConfig& config, int layer);
int num_gdn_layers(const ModelConfig& config);
int gdn_layer_index(const ModelConfig& config, int layer);

}  // namespace qwen35
}  // namespace ccinfer
