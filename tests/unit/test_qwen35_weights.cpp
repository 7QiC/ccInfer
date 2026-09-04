#include <filesystem>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "backend/backend.h"
#include "checkpoint/checkpoint.h"
#include "model/qwen35/qwen35_weights.h"

namespace ccinfer {
namespace {

#ifndef GGUF_ARTIFACT_PATH
#define GGUF_ARTIFACT_PATH "models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf"
#endif

TEST(Qwen35WeightsTest, LoadsRealGgufShapesAndTiedLmHead) {
    const std::string gguf_path = GGUF_ARTIFACT_PATH;
    if (!std::filesystem::exists(gguf_path)) {
        GTEST_SKIP() << "Qwen3.5 GGUF artifact not present";
    }
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    auto checkpoint_r = Checkpoint::open(gguf_path);
    ASSERT_TRUE(checkpoint_r.has_value());
    auto& checkpoint = **checkpoint_r;

    auto config_r = checkpoint.load_config();
    ASSERT_TRUE(config_r.has_value());
    const auto& config = *config_r;

    auto weights_r = Qwen35Weights::load(backend, config, checkpoint.weights());
    ASSERT_TRUE(weights_r.has_value());
    const auto& w = *weights_r;

    EXPECT_EQ(config.arch_, ModelArch::Qwen3_5);
    EXPECT_EQ(config.d_model_, 2048);
    EXPECT_EQ(w.embed.shape(0), config.vocab_size_);
    EXPECT_EQ(w.embed.shape(1), config.d_model_);
    EXPECT_EQ(w.embed.dtype(), ccop::DType::kBFloat16);
    EXPECT_EQ(w.lm_head.data(), w.embed.data());
    EXPECT_EQ(w.lm_head.shape(0), config.vocab_size_);
    EXPECT_EQ(w.lm_head.shape(1), config.d_model_);
    EXPECT_EQ(w.gdn_layers_.size(), 18u);
    EXPECT_EQ(w.attn_layers_.size(), 6u);

    const auto& gdn0 = w.gdn_layers_[0];
    EXPECT_EQ(gdn0.attn_qkv.shape(0), 6144);
    EXPECT_EQ(gdn0.attn_qkv.shape(1), 2048);
    EXPECT_EQ(gdn0.attn_gate.shape(0), 2048);
    EXPECT_EQ(gdn0.attn_gate.shape(1), 2048);
    EXPECT_EQ(gdn0.ssm_conv1d.shape(0), 6144);
    EXPECT_EQ(gdn0.ssm_conv1d.shape(1), 4);
    EXPECT_EQ(gdn0.ssm_a.shape(0), 16);
    EXPECT_EQ(gdn0.ssm_a.dtype(), ccop::DType::kFloat32);
    EXPECT_EQ(gdn0.ssm_dt_bias.dtype(), ccop::DType::kFloat32);

    const auto& attn0 = w.attn_layers_[0];
    EXPECT_EQ(attn0.attn_q.shape(0), 2048);
    EXPECT_EQ(attn0.attn_q.shape(1), 2048);
    EXPECT_EQ(attn0.attn_gate.shape(0), 2048);
    EXPECT_EQ(attn0.attn_gate.shape(1), 2048);
    EXPECT_EQ(attn0.attn_k.shape(0), 512);
    EXPECT_EQ(attn0.attn_k.shape(1), 2048);
}

}  // namespace
}  // namespace ccinfer
