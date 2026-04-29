#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "model/config.h"

using namespace ccinfer;
using namespace ccinfer::engine;

TEST(ModelConfigTest, FromLlamaJson) {
    nlohmann::json j = {{"architectures", {"LlamaForCausalLM"}},
                        {"hidden_size", 2048},
                        {"num_attention_heads", 16},
                        {"num_key_value_heads", 4},
                        {"num_hidden_layers", 16},
                        {"intermediate_size", 5632},
                        {"vocab_size", 128256},
                        {"max_position_embeddings", 32768},
                        {"rope_theta", 500000.0},
                        {"rms_norm_eps", 1e-5}};

    auto cfg = ModelConfig::from_json(j);
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->arch, ModelArch::Llama);
    EXPECT_EQ(cfg->n_layers, 16);
    EXPECT_EQ(cfg->n_q_heads, 16);
    EXPECT_EQ(cfg->n_kv_heads, 4);
    EXPECT_EQ(cfg->d_model, 2048);
    EXPECT_EQ(cfg->d_ff, 5632);
    EXPECT_EQ(cfg->vocab_size, 128256);
    EXPECT_EQ(cfg->max_seq_len, 32768);
    EXPECT_EQ(cfg->rope_theta, 500000.0f);
    EXPECT_FLOAT_EQ(cfg->rms_norm_eps, 1e-5f);
}

TEST(ModelConfigTest, UnknownArchitecture) {
    nlohmann::json j = {{"architectures", {"UnknownModel"}}, {"hidden_size", 1024},
                        {"num_attention_heads", 8},          {"num_hidden_layers", 8},
                        {"intermediate_size", 2048},         {"vocab_size", 50000},
                        {"max_position_embeddings", 4096}};

    auto cfg = ModelConfig::from_json(j);
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->arch, ModelArch::Unknown);
}

TEST(ModelConfigTest, MissingRequiredFields) {
    nlohmann::json j = {{"architectures", {"LlamaForCausalLM"}}};
    auto cfg = ModelConfig::from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelLoadFailed);
}
