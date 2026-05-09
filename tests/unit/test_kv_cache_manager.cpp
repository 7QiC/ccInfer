#include <gtest/gtest.h>

#include "engine/cache/block.h"
#include "engine/cache/kv_cache_manager.h"

namespace ccinfer {
namespace engine {
namespace {

class KVCacheManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto r = mgr_.init(64);
        ASSERT_TRUE(r.has_value());
    }
    KVCacheManager mgr_;
};

TEST_F(KVCacheManagerTest, InitState) {
    EXPECT_EQ(mgr_.num_free_blocks(), 64);
    EXPECT_EQ(mgr_.max_blocks(), 64);
}

TEST_F(KVCacheManagerTest, AllocateSingleBlock) {
    auto r = mgr_.allocate_blocks(1);
    ASSERT_TRUE(r.has_value());

    EXPECT_EQ(r->size(), 1);
    EXPECT_EQ(mgr_.num_free_blocks(), 63);
}

TEST_F(KVCacheManagerTest, AllocateMultipleBlocks) {
    auto r = mgr_.allocate_blocks(3);
    ASSERT_TRUE(r.has_value());

    EXPECT_EQ(r->size(), 3);
    EXPECT_EQ(mgr_.num_free_blocks(), 61);
}

TEST_F(KVCacheManagerTest, ReleaseReturnsBlocksToFreeList) {
    auto r = mgr_.allocate_blocks(2);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(mgr_.num_free_blocks(), 62);

    auto rel = mgr_.release_blocks(*r);
    ASSERT_TRUE(rel.has_value());
    EXPECT_EQ(mgr_.num_free_blocks(), 64);
}

TEST_F(KVCacheManagerTest, DoubleFreeDetected) {
    auto r = mgr_.allocate_blocks(1);
    ASSERT_TRUE(r.has_value());

    ASSERT_TRUE(mgr_.release_blocks(*r).has_value());
    auto rel2 = mgr_.release_blocks(*r);
    EXPECT_FALSE(rel2.has_value());
    EXPECT_EQ(rel2.error(), ErrorCode::KVBlockDoubleFree);
}

TEST_F(KVCacheManagerTest, Exhaustion) {
    for (int i = 0; i < 64; ++i) {
        auto r = mgr_.allocate_blocks(1);
        ASSERT_TRUE(r.has_value()) << "exhaustion at block " << i;
    }
    EXPECT_EQ(mgr_.num_free_blocks(), 0);

    auto r = mgr_.allocate_blocks(1);
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::KVBlockExhausted);
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
