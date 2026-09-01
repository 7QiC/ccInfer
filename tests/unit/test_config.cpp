#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "checkpoint/huggingface/config_loader.h"
#include "config/engine_config.h"
#include "config/model_config.h"

using namespace ccinfer;

TEST(HfConfigLoaderTest, FromQwen3Json) {
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

    auto cfg = HfConfigLoader::load_from_json(j);
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

TEST(HfConfigLoaderTest, AcceptsExplicitHeadDimDifferentFromHiddenSize) {
    // Qwen3-0.6B style: n_q_heads * head_dim != hidden_size is valid because
    // Qwen3 projections are sized by (n_heads * head_dim), not hidden_size.
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}},
                        {"hidden_size", 1024},
                        {"num_attention_heads", 16},
                        {"num_key_value_heads", 8},
                        {"num_hidden_layers", 28},
                        {"intermediate_size", 3072},
                        {"vocab_size", 151936},
                        {"max_position_embeddings", 40960},
                        {"head_dim", 128},
                        {"rope_theta", 1000000.0},
                        {"rms_norm_eps", 1e-6}};

    auto cfg = HfConfigLoader::load_from_json(j);
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->d_model_, 1024);
    EXPECT_EQ(cfg->n_q_heads_, 16);
    EXPECT_EQ(cfg->head_dim_, 128);
}

TEST(HfConfigLoaderTest, UnknownArchitecture) {
    nlohmann::json j = {{"architectures", {"UnknownModel"}}, {"hidden_size", 1024},
                        {"num_attention_heads", 8},          {"num_hidden_layers", 8},
                        {"intermediate_size", 2048},         {"vocab_size", 50000},
                        {"max_position_embeddings", 4096}};

    auto cfg = HfConfigLoader::load_from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelUnsupportedArch);
}

TEST(HfConfigLoaderTest, MissingRequiredFields) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}}};
    auto cfg = HfConfigLoader::load_from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelConfigInvalid);
}

TEST(HfConfigLoaderTest, RejectsNonDivisibleHiddenSize) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}},
                        {"hidden_size", 2049},
                        {"num_attention_heads", 16},
                        {"num_hidden_layers", 16},
                        {"intermediate_size", 5632},
                        {"vocab_size", 128256}};

    auto cfg = HfConfigLoader::load_from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelConfigInvalid);
}

TEST(HfConfigLoaderTest, RejectsInvalidHeadTopology) {
    nlohmann::json j = {{"architectures", {"Qwen3ForCausalLM"}},
                        {"hidden_size", 2048},
                        {"num_attention_heads", 16},
                        {"num_key_value_heads", 32},
                        {"num_hidden_layers", 16},
                        {"intermediate_size", 5632},
                        {"vocab_size", 128256}};

    auto cfg = HfConfigLoader::load_from_json(j);
    EXPECT_FALSE(cfg.has_value());
    EXPECT_EQ(cfg.error(), ErrorCode::ModelConfigInvalid);
}

TEST(HfConfigLoaderTest, LoadsFromConfigJsonPath) {
    const auto model_dir = std::filesystem::path("/tmp") / "ccinfer-hf-config-test";
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

    auto result = HfConfigLoader::load((model_dir / "config.json").string());
    cleanup();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->n_layers_, 16);
    EXPECT_EQ(result->max_seq_len_, 2048);
}

TEST(ModelConfigTest, ValidateAcceptsCanonicalQwen3) {
    ModelConfig config = HfConfigLoader::load_from_json(
                             nlohmann::json{{"architectures", {"Qwen3ForCausalLM"}},
                                            {"hidden_size", 1024},
                                            {"num_attention_heads", 16},
                                            {"num_key_value_heads", 8},
                                            {"num_hidden_layers", 28},
                                            {"intermediate_size", 3072},
                                            {"vocab_size", 151936},
                                            {"max_position_embeddings", 4096}})
                             .value();
    EXPECT_TRUE(config.validate().has_value());
}

TEST(ModelConfigTest, ValidateRejectsZeroHeadDim) {
    ModelConfig config;
    config.arch_ = ModelArch::Qwen3;
    config.n_layers_ = 1;
    config.n_q_heads_ = 4;
    config.n_kv_heads_ = 2;
    config.d_model_ = 64;
    config.head_dim_ = 0;
    config.d_ff_ = 128;
    config.vocab_size_ = 100;
    config.max_seq_len_ = 32;
    EXPECT_FALSE(config.validate().has_value());
}

TEST(EngineConfigTest, DefaultKvBlockSizeIs128) {
    EngineConfig config;
    EXPECT_EQ(config.kv_block_size, 128);
    EXPECT_TRUE(config.validate().has_value());
}

TEST(EngineConfigTest, Accepts64_128_256KvBlockSizes) {
    for (int block_size : {64, 128, 256}) {
        EngineConfig config;
        config.kv_block_size = block_size;
        config.max_blocks = 4;
        config.default_max_context_len = 128;
        auto result = config.validate();
        EXPECT_TRUE(result.has_value()) << "kv_block_size=" << block_size;
    }
}

TEST(EngineConfigTest, RejectsNonPositiveKvBlockSize) {
    for (int block_size : {0, -1}) {
        EngineConfig config;
        config.kv_block_size = block_size;
        auto result = config.validate();
        EXPECT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
    }
}

TEST(EngineConfigTest, RejectsContextLargerThanCache) {
    EngineConfig config;
    config.max_blocks = 2;
    config.kv_block_size = 16;
    config.default_max_context_len = 33;

    auto result = config.validate();
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), ErrorCode::InvalidArgument);
}

TEST(ConfigBoundaryTest, MaxSeqLenIsDistinctFromRuntimeContextLimit) {
    auto model = HfConfigLoader::load_from_json(
                     nlohmann::json{{"architectures", {"Qwen3ForCausalLM"}},
                                    {"hidden_size", 2048},
                                    {"num_attention_heads", 16},
                                    {"num_key_value_heads", 4},
                                    {"num_hidden_layers", 16},
                                    {"intermediate_size", 5632},
                                    {"vocab_size", 128256},
                                    {"max_position_embeddings", 4096}})
                     .value();
    EngineConfig engine;

    EXPECT_EQ(model.max_seq_len_, 4096);
    EXPECT_EQ(engine.default_max_context_len, 2048);
    EXPECT_NE(model.max_seq_len_, engine.default_max_context_len);
}
