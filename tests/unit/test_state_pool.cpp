#include <optional>

#include <gtest/gtest.h>

#include "backend/backend.h"
#include "config/model_config.h"
#include "state/state_pool.h"

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

TEST(StateTest, DefaultResourceMetadata) {
    State state;
    EXPECT_EQ(state.id, kInvalidStateId);
    EXPECT_EQ(state.kind, StateKind::Mutable);
    EXPECT_TRUE(state.is_free());
    EXPECT_FALSE(state.is_in_use());

    state.kind = StateKind::Snapshot;
    state.status = StateStatus::InUse;
    EXPECT_TRUE(state.is_snapshot_kind());
    EXPECT_TRUE(state.is_in_use());
    EXPECT_FALSE(state.is_free());
}

TEST(StateTest, FreeListPushPop) {
    StateFreeList fl;
    State states[3];
    for (int i = 0; i < 3; ++i) states[i].id = i;

    fl.push_back(states[0]);
    fl.push_back(states[1]);
    fl.push_back(states[2]);
    EXPECT_EQ(static_cast<int>(fl.size()), 3);

    auto& front = fl.front();
    fl.pop_front();
    EXPECT_EQ(front.id, 0);
    EXPECT_EQ(static_cast<int>(fl.size()), 2);

    fl.pop_front();
    fl.pop_front();
}

TEST(StatePoolTest, ActiveAcquireReleaseReusesStates) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 3);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(20).has_value());
    EXPECT_EQ(pool.active_state_of(10), 0);
    EXPECT_EQ(pool.active_state_of(20), 1);
    EXPECT_FALSE(pool.active_state_of(30).has_value());

    // Same-sequence multiple outstanding batches share the same active state.
    ASSERT_TRUE(pool.acquire_active(10).has_value());
    EXPECT_EQ(pool.active_state_of(10), 0);

    ASSERT_TRUE(pool.release_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(30).has_value());
    EXPECT_EQ(pool.active_state_of(30), 0);
    EXPECT_FALSE(pool.active_state_of(10).has_value());
}

TEST(StatePoolTest, ActiveStateIsNotReusedUntilExplicitRelease) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 1, 0);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    EXPECT_EQ(pool.active_state_of(10), 0);
    EXPECT_FALSE(pool.acquire_active(20).has_value());

    ASSERT_TRUE(pool.release_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(20).has_value());
    EXPECT_EQ(pool.active_state_of(20), 0);
}

TEST(StatePoolTest, SnapshotResourcesAreSeparateAndReusable) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 2);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;
    EXPECT_EQ(pool.max_active(), 2);
    EXPECT_EQ(pool.max_snapshots(), 2);
    EXPECT_EQ(pool.num_free_snapshots(), 2);

    auto first = pool.acquire_snapshot();
    auto second = pool.acquire_snapshot();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(*first, 2);
    EXPECT_EQ(*second, 3);
    EXPECT_EQ(pool.num_free_snapshots(), 0);
    EXPECT_FALSE(pool.acquire_snapshot().has_value());

    ASSERT_TRUE(pool.release_snapshot(*first).has_value());
    EXPECT_EQ(pool.num_free_snapshots(), 1);
    auto reused = pool.acquire_snapshot();
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(*reused, *first);
}

TEST(StatePoolTest, ZeroSnapshotCapacityRejectsAcquire) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 0);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;
    EXPECT_EQ(pool.num_free_snapshots(), 0);
    EXPECT_FALSE(pool.acquire_snapshot().has_value());
}

}  // namespace
}  // namespace ccinfer
