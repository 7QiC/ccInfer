#include <gtest/gtest.h>

#include "backend/backend.h"
#include "cache/state_cache.h"
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

TEST(StateCacheTest, SnapshotRestoreAndLruTouch) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 2);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    StateCache cache(pool, /*frontier_depth=*/2);
    ASSERT_TRUE(pool.acquire_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(20).has_value());
    const StateId active0 = *pool.active_state_of(10);
    const StateId active1 = *pool.active_state_of(20);

    constexpr uint64_t kHashA = 0xAAAA000000000001ULL;
    constexpr uint64_t kHashB = 0xBBBB000000000001ULL;
    ASSERT_TRUE(cache.snapshot(kHashA, 0, active0).has_value());
    ASSERT_TRUE(cache.snapshot(kHashB, 1, active0).has_value());
    ASSERT_TRUE(cache.has(kHashA));
    ASSERT_TRUE(cache.has(kHashB));
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_NE(cache.state_id_of(kHashA), kInvalidStateId);
    EXPECT_EQ(pool.num_free_snapshots(), 0);

    // Restore is prefix-keyed, not seq-keyed.
    ASSERT_TRUE(cache.restore(kHashA, active1).has_value());
    // Snapshot exists; duplicate snapshot is a no-op and does not consume a slot.
    ASSERT_TRUE(cache.snapshot(kHashA, 0, active1).has_value());
    EXPECT_EQ(cache.size(), 2u);
}

TEST(StateCacheTest, FrontierDepthRejectsOutOfRangeSnapshot) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 1, 2);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;
    StateCache cache(pool, /*frontier_depth=*/2);

    ASSERT_TRUE(pool.acquire_active(7).has_value());
    const StateId active = *pool.active_state_of(7);
    EXPECT_FALSE(cache.snapshot(0x1, 2, active).has_value());
    EXPECT_FALSE(cache.snapshot(0x1, -1, active).has_value());
    EXPECT_FALSE(cache.restore(0x1, active).has_value());
}

TEST(StateCacheTest, EvictionReleasesSnapshotAndRecyclesResource) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 1, 2);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;
    StateCache cache(pool, /*frontier_depth=*/2);

    ASSERT_TRUE(pool.acquire_active(7).has_value());
    const StateId active = *pool.active_state_of(7);

    constexpr uint64_t kHashA = 0xAAAA000000000001ULL;
    constexpr uint64_t kHashB = 0xBBBB000000000001ULL;
    ASSERT_TRUE(cache.snapshot(kHashA, 0, active).has_value());
    ASSERT_TRUE(cache.snapshot(kHashB, 1, active).has_value());
    EXPECT_EQ(pool.num_free_snapshots(), 0);

    // Restore A so B becomes LRU, then a third snapshot must evict B.
    ASSERT_TRUE(cache.restore(kHashA, active).has_value());
    constexpr uint64_t kHashC = 0xCCCC000000000001ULL;
    ASSERT_TRUE(cache.snapshot(kHashC, 0, active).has_value());
    EXPECT_TRUE(cache.has(kHashC));
    EXPECT_TRUE(cache.has(kHashA));
    EXPECT_FALSE(cache.has(kHashB));
    EXPECT_EQ(cache.size(), 2u);
    EXPECT_EQ(pool.num_free_snapshots(), 0);
    EXPECT_GT(cache.stats().evictions, 0u);
}

TEST(StateCacheTest, ZeroDepthDisablesSnapshots) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 1, 0);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;
    StateCache cache(pool, /*frontier_depth=*/0);

    ASSERT_TRUE(pool.acquire_active(7).has_value());
    const StateId active = *pool.active_state_of(7);
    EXPECT_FALSE(cache.snapshot(1, 0, active).has_value());
    EXPECT_FALSE(cache.restore(1, active).has_value());
}

}  // namespace
}  // namespace ccinfer
