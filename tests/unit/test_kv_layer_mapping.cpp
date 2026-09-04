#include <vector>

#include <gtest/gtest.h>

#include "config/model_config.h"
#include "model/qwen35/qwen35_model.h"

namespace ccinfer {
namespace {

ModelConfig qwen35_config() {
    ModelConfig cfg;
    cfg.arch_ = ModelArch::Qwen3_5;
    cfg.n_layers_ = 24;
    cfg.layer_types_.reserve(25);
    for (int i = 0; i < 24; ++i) {
        cfg.layer_types_.push_back((i % 4 == 3) ? LayerType::FullAttention
                                                : LayerType::GatedDeltaNet);
    }
    cfg.layer_types_.push_back(LayerType::MtpPredictor);
    return cfg;
}

TEST(KvLayerMappingTest, Qwen35CountsAndMapping) {
    const auto cfg = qwen35_config();
    EXPECT_EQ(qwen35::num_kv_layers(cfg), 6);
    EXPECT_EQ(qwen35::num_gdn_layers(cfg), 18);

    const std::vector<int> kv_layers = {3, 7, 11, 15, 19, 23};
    for (std::size_t i = 0; i < kv_layers.size(); ++i) {
        EXPECT_EQ(qwen35::kv_layer_index(cfg, kv_layers[i]), static_cast<int>(i));
    }

    EXPECT_EQ(qwen35::gdn_layer_index(cfg, 0), 0);
    EXPECT_EQ(qwen35::gdn_layer_index(cfg, 4), 3);
    EXPECT_EQ(qwen35::gdn_layer_index(cfg, 22), 17);
}

}  // namespace
}  // namespace ccinfer
