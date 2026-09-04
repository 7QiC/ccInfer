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

TEST(StatePoolTest, ActiveAcquireReleaseReusesSlots) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 3, 3);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(20).has_value());
    EXPECT_EQ(pool.active_slot_of(10), 0);
    EXPECT_EQ(pool.active_slot_of(20), 1);
    EXPECT_FALSE(pool.active_slot_of(30).has_value());

    // Same-sequence multiple outstanding batches share the same active slot.
    ASSERT_TRUE(pool.acquire_active(10).has_value());
    EXPECT_EQ(pool.active_slot_of(10), 0);

    ASSERT_TRUE(pool.release_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(30).has_value());
    EXPECT_EQ(pool.active_slot_of(30), 0);
    EXPECT_FALSE(pool.active_slot_of(10).has_value());
}

TEST(StatePoolTest, CachedIdentityIsPrefixHashNotSeqId) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 3, 3);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(20).has_value());
    const StateSlotId a_slot = *pool.active_slot_of(10);
    const StateSlotId b_slot = *pool.active_slot_of(20);

    // prefixA at three block frontiers (hashes are arbitrary but distinct in
    // this unit test; real callers use PrefixCache::chain_hashes).
    constexpr uint64_t kPrefixA0 = 0xA000000000000001ULL;
    constexpr uint64_t kPrefixA1 = 0xA000000000000002ULL;
    constexpr uint64_t kPrefixA2 = 0xA000000000000003ULL;
    ASSERT_TRUE(pool.snapshot(kPrefixA0, 0, a_slot).has_value());
    ASSERT_TRUE(pool.snapshot(kPrefixA1, 1, a_slot).has_value());
    ASSERT_TRUE(pool.snapshot(kPrefixA2, 2, a_slot).has_value());

    // seq B can restore the same prefix identity; seq_id is not part of the key.
    EXPECT_TRUE(pool.has_cached(kPrefixA0));
    EXPECT_TRUE(pool.has_cached(kPrefixA1));
    EXPECT_TRUE(pool.has_cached(kPrefixA2));
    EXPECT_FALSE(pool.has_cached(kPrefixA0 + 1));
    ASSERT_TRUE(pool.restore(kPrefixA1, b_slot).has_value());
    EXPECT_NE(pool.cached_slot_of(kPrefixA1), kInvalidStateSlot);
    EXPECT_FALSE(pool.snapshot(kPrefixA0, 3, a_slot).has_value());  // depth 3.
}

TEST(StatePoolTest, FrontierDepthIsSeparateFromCachedCapacity) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    // depth=3, capacity=2: capacity can fill with two distinct prefixes while
    // both are at frontier 0, proving they are not the same concept.
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 3, 2);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    const StateSlotId slot = *pool.active_slot_of(10);
    constexpr uint64_t kHashA = 0xB000000000000001ULL;
    constexpr uint64_t kHashB = 0xB000000000000002ULL;
    constexpr uint64_t kHashC = 0xB000000000000003ULL;

    ASSERT_TRUE(pool.snapshot(kHashA, 0, slot).has_value());
    ASSERT_TRUE(pool.snapshot(kHashB, 0, slot).has_value());
    // Capacity is exhausted although frontier depth still permits more.
    EXPECT_FALSE(pool.snapshot(kHashC, 0, slot).has_value());
    EXPECT_TRUE(pool.has_cached(kHashA));
    EXPECT_TRUE(pool.has_cached(kHashB));

    // Frontier depth still rejects a frontier >= 3 even with free capacity.
    auto pool2_r = StatePool::create(**backend_r, qwen35_config(), 2, 3, 5);
    ASSERT_TRUE(pool2_r.has_value());
    auto& pool2 = **pool2_r;
    ASSERT_TRUE(pool2.acquire_active(20).has_value());
    const StateSlotId slot2 = *pool2.active_slot_of(20);
    ASSERT_TRUE(pool2.snapshot(kHashA, 0, slot2).has_value());
    EXPECT_FALSE(pool2.snapshot(kHashB, 3, slot2).has_value());
}

TEST(StatePoolTest, ActiveSlotIsNotReusedUntilExplicitRelease) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    // max_active=1 makes reuse observable: A keeps slot 0, B must not steal it.
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 1, 0, 0);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    EXPECT_EQ(pool.active_slot_of(10), 0);
    EXPECT_FALSE(pool.acquire_active(20).has_value());

    ASSERT_TRUE(pool.release_active(10).has_value());
    ASSERT_TRUE(pool.acquire_active(20).has_value());
    EXPECT_EQ(pool.active_slot_of(20), 0);
}

TEST(StatePoolTest, ZeroCacheDepthDisablesSnapshot) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto pool_r = StatePool::create(**backend_r, qwen35_config(), 2, 0, 0);
    ASSERT_TRUE(pool_r.has_value());
    auto& pool = **pool_r;

    ASSERT_TRUE(pool.acquire_active(10).has_value());
    const StateSlotId slot = *pool.active_slot_of(10);
    EXPECT_FALSE(pool.snapshot(1, 0, slot).has_value());
    EXPECT_FALSE(pool.restore(1, slot).has_value());
}

}  // namespace
}  // namespace ccinfer
