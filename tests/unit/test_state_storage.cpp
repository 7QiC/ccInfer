#include <gtest/gtest.h>

#include "backend/backend.h"
#include "state/state_storage.h"
#include "config/model_config.h"

namespace ccinfer {
namespace {

ModelConfig qwen35_config() {
    ModelConfig cfg;
    cfg.arch_ = ModelArch::Qwen3_5;
    cfg.n_layers_ = 24;
    cfg.n_q_heads_ = 8;
    cfg.n_kv_heads_ = 2;
    cfg.d_model_ = 2048;
    cfg.head_dim_ = 256;
    cfg.rotary_dim_ = 64;
    cfg.d_ff_ = 6144;
    cfg.vocab_size_ = 248320;
    cfg.max_seq_len_ = 4096;
    cfg.rope_theta_ = 10000000.0f;
    cfg.rms_norm_eps_ = 1e-6f;
    cfg.full_attention_interval_ = 4;
    cfg.nextn_predict_layers_ = 1;
    cfg.ssm_conv_kernel_ = 4;
    cfg.ssm_state_size_ = 128;
    cfg.ssm_group_count_ = 16;
    cfg.ssm_time_step_rank_ = 16;
    cfg.ssm_inner_size_ = 2048;
    cfg.layer_types_.reserve(25);
    for (int i = 0; i < 24; ++i) {
        cfg.layer_types_.push_back(i % 4 == 3 ? LayerType::FullAttention
                                              : LayerType::GatedDeltaNet);
    }
    cfg.layer_types_.push_back(LayerType::MtpPredictor);
    return cfg;
}

TEST(StateStorageTest, PreallocatesZeroedSlotsAndCopies) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";

    const auto cfg = qwen35_config();
    auto storage_r = StateStorage::create(**backend_r, cfg, 5);
    ASSERT_TRUE(storage_r.has_value());
    auto& storage = **storage_r;

    EXPECT_EQ(storage.num_slots(), 5);
    EXPECT_EQ(storage.recurrent_state(0).shape(0), 5);
    EXPECT_EQ(storage.recurrent_state(0).shape(1), 16);
    EXPECT_EQ(storage.recurrent_state(0).shape(2), 128);
    EXPECT_EQ(storage.recurrent_state(0).shape(3), 128);
    EXPECT_EQ(storage.conv_state(0).shape(0), 5);
    EXPECT_EQ(storage.conv_state(0).shape(1), 6144);
    EXPECT_EQ(storage.conv_state(0).shape(2), 3);

    ASSERT_TRUE(storage.zero_slot(0).has_value());
    ASSERT_TRUE(storage.copy_slot(0, 1).has_value());
    ASSERT_TRUE(storage.zero_slot(2).has_value());
    ASSERT_TRUE(storage.copy_slot(2, 4).has_value());
    EXPECT_FALSE(storage.copy_slot(4, 5).has_value());
    EXPECT_FALSE(storage.zero_slot(-1).has_value());
}

}  // namespace
}  // namespace ccinfer
