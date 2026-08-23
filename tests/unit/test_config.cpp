#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "config/config.h"
#include "model/config.h"

using namespace ccinfer;

TEST(ModelConfigTest, FromQwen3Json) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}},
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
    EXPECT_EQ(cfg->arch_, ModelArch::Qwen3);
    EXPECT_EQ(cfg->n_layers_, 16);
    EXPECT_EQ(cfg->n_q_heads_, 16);
    EXPECT_EQ(cfg->n_kv_heads_, 4);
    EXPECT_EQ(cfg->d_model_, 2048);
    EXPECT_EQ(cfg->d_ff_, 5632);
    EXPECT_EQ(cfg->vocab_size_, 128256);
    EXPECT_EQ(cfg->max_seq_len_, 32768);
    EXPECT_EQ(cfg->rope_theta_, 500000.0f);
    EXPECT_FLOAT_EQ(cfg->rms_norm_eps_, 1e-5f);
}

TEST(ModelConfigTest, UnknownArchitecture) {
    nlohmann::json j = {{"architectures", {"UnknownModel"}}, {"hidden_size", 1024},
                        {"num_attention_heads", 8},          {"num_hidden_layers", 8},
                        {"intermediate_size", 2048},         {"vocab_size", 50000},
                        {"max_position_embeddings", 4096}};

    auto cfg = ModelConfig::from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelUnsupportedArch);
}

TEST(ModelConfigTest, MissingRequiredFields) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}}};
    auto cfg = ModelConfig::from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelConfigInvalid);
}

TEST(ModelConfigTest, RejectsNonDivisibleHiddenSize) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}},
                        {"hidden_size", 2049},
                        {"num_attention_heads", 16},
                        {"num_hidden_layers", 16},
                        {"intermediate_size", 5632},
                        {"vocab_size", 128256}};

    auto cfg = ModelConfig::from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelConfigInvalid);
}

TEST(ModelConfigTest, RejectsInvalidHeadTopology) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}},
                        {"hidden_size", 2048},
                        {"num_attention_heads", 16},
                        {"num_key_value_heads", 32},
                        {"num_hidden_layers", 16},
                        {"intermediate_size", 5632},
                        {"vocab_size", 128256}};

    auto cfg = ModelConfig::from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelConfigInvalid);
}

TEST(EngineConfigTest, RejectsContextLargerThanCache) {
    EngineConfig config;
    config.max_blocks = 2;
    config.block_size = 16;
    config.default_max_context_len = 33;

    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

TEST(ConfigTest, LoadsModelAndEngineConfigTogether) {
    const auto model_dir = std::filesystem::path("/tmp") / "ccinfer-config-test";
    std::filesystem::create_directories(model_dir);
    const auto cleanup = [&] { std::filesystem::remove_all(model_dir); };

    std::ofstream config_file(model_dir / "config.json");
    ASSERT_TRUE(config_file.is_open());
    config_file << R"({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_hidden_layers": 16,
        "intermediate_size": 5632,
        "vocab_size": 128256
    })";
    config_file.close();

    EngineConfig engine;
    engine.max_blocks = 128;
    engine.default_max_context_len = 1024;
    auto result = Config::load(model_dir.string(), engine);
    cleanup();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->model_path_, model_dir.string());
    EXPECT_EQ(result->model_.n_layers_, 16);
    EXPECT_EQ(result->engine_.max_blocks, 128);
    EXPECT_EQ(result->engine_.default_max_context_len, 1024);
}
