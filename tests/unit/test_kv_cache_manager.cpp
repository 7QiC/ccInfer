#include <gtest/gtest.h>

#include "engine/cache/block.h"
#include "engine/cache/kv_cache_manager.h"
#include "engine/model/config.h"

namespace ccinfer {
namespace engine {
namespace {

ModelConfig make_test_config() {
    ModelConfig cfg;
    cfg.n_layers_ = 4;
    cfg.n_q_heads_ = 8;
    cfg.n_kv_heads_ = 8;
    cfg.d_model_ = 1024;
    cfg.head_dim_ = 128;
    cfg.d_ff_ = 4096;
    cfg.vocab_size_ = 32000;
    return cfg;
}

class KVCacheManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto r = mgr_.init(make_test_config(), 64);
        ASSERT_TRUE(r.has_value());
    }
    KVCacheManager mgr_;
};

TEST_F(KVCacheManagerTest, InitState) {
    EXPECT_EQ(mgr_.num_free_blocks(), 64);
    EXPECT_EQ(mgr_.num_layers(), 4);
    EXPECT_EQ(mgr_.block_size(), kKVBlockSize);
}

TEST_F(KVCacheManagerTest, AllocateSingleBlock) {
    std::vector<int32_t> tokens(8, 1);
    auto r = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(r.has_value());

    EXPECT_EQ(r->block_table.size(), 1);
    EXPECT_EQ(mgr_.num_free_blocks(), 63);

    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(r->slot_mapping[i], i);  // block_id=0 * 16 + i
    }
}

TEST_F(KVCacheManagerTest, AllocateMultipleBlocks) {
    std::vector<int32_t> tokens(40, 1);  // ceil(40/16) = 3 blocks
    auto r = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(r.has_value());

    EXPECT_EQ(r->block_table.size(), 3);
    EXPECT_EQ(mgr_.num_free_blocks(), 61);

    // Block 0: tokens 0..15
    int32_t b0 = r->block_table[0];
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(r->slot_mapping[i], b0 * kKVBlockSize + i);
    }

    // Block 1: tokens 16..31
    int32_t b1 = r->block_table[1];
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(r->slot_mapping[16 + i], b1 * kKVBlockSize + i);
    }

    // Block 2: tokens 32..39 (partial)
    int32_t b2 = r->block_table[2];
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(r->slot_mapping[32 + i], b2 * kKVBlockSize + i);
    }
}

TEST_F(KVCacheManagerTest, ReleaseReturnsBlocksToFreeList) {
    std::vector<int32_t> tokens(20, 1);
    auto r = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(mgr_.num_free_blocks(), 62);

    auto rel = mgr_.release_blocks(r->block_table);
    ASSERT_TRUE(rel.has_value());
    EXPECT_EQ(mgr_.num_free_blocks(), 64);
}

TEST_F(KVCacheManagerTest, DoubleFreeDetected) {
    std::vector<int32_t> tokens(8, 1);
    auto r = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(r.has_value());

    ASSERT_TRUE(mgr_.release_blocks(r->block_table).has_value());
    auto rel2 = mgr_.release_blocks(r->block_table);
    EXPECT_FALSE(rel2.has_value());
    EXPECT_EQ(rel2.error(), ErrorCode::KVBlockDoubleFree);
}

TEST_F(KVCacheManagerTest, Exhaustion) {
    for (int i = 0; i < 64; ++i) {
        std::vector<int32_t> tokens(kKVBlockSize, 1);
        auto r = mgr_.prepare_blocks(tokens);
        ASSERT_TRUE(r.has_value()) << "exhaustion at block " << i;
    }
    EXPECT_EQ(mgr_.num_free_blocks(), 0);

    auto r = mgr_.prepare_blocks({1});
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::KVBlockExhausted);
}

TEST_F(KVCacheManagerTest, CachePointersNonNull) {
    EXPECT_NE(mgr_.k_cache(0), nullptr);
    EXPECT_NE(mgr_.v_cache(0), nullptr);
    EXPECT_NE(mgr_.k_cache(0), mgr_.k_cache(1));
    EXPECT_NE(mgr_.v_cache(0), mgr_.v_cache(1));
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
