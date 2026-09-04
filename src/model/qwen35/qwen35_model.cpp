#include "model/qwen35/qwen35_model.h"

#include <cassert>

namespace ccinfer {
namespace qwen35 {

namespace {
int count_layer_type(const ModelConfig& config, LayerType type) {
    assert(config.arch_ == ModelArch::Qwen3_5);
    int count = 0;
    const int main_layers = config.n_layers_;
    const int declared = static_cast<int>(config.layer_types_.size());
    for (int i = 0; i < main_layers && i < declared; ++i) {
        if (config.layer_types_[static_cast<std::size_t>(i)] == type) ++count;
    }
    return count;
}

int index_for_layer_type(const ModelConfig& config, int layer, LayerType type) {
    assert(config.arch_ == ModelArch::Qwen3_5);
    assert(layer >= 0 && layer < config.n_layers_);
    assert(layer < static_cast<int>(config.layer_types_.size()));
    assert(config.layer_types_[static_cast<std::size_t>(layer)] == type);
    int index = 0;
    for (int i = 0; i < layer; ++i) {
        if (config.layer_types_[static_cast<std::size_t>(i)] == type) ++index;
    }
    return index;
}

}  // namespace

int num_kv_layers(const ModelConfig& config) {
    return count_layer_type(config, LayerType::FullAttention);
}

int kv_layer_index(const ModelConfig& config, int layer) {
    return index_for_layer_type(config, layer, LayerType::FullAttention);
}

int num_gdn_layers(const ModelConfig& config) {
    return count_layer_type(config, LayerType::GatedDeltaNet);
}

int gdn_layer_index(const ModelConfig& config, int layer) {
    return index_for_layer_type(config, layer, LayerType::GatedDeltaNet);
}

}  // namespace qwen35
}  // namespace ccinfer
