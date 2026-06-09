#include <gtest/gtest.h>

#include <cuda_bf16.h>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/cache/kv_cache_manager.h"
#include "engine/cache/kv_cache_storage.h"

namespace ccinfer {
namespace engine {
namespace {

class KVCacheManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto be = CudaBackend::create(0);
        ASSERT_TRUE(be.has_value());
        backend_ = std::move(*be);

        auto sr = KVCacheStorage::create<__nv_bfloat16>(*backend_, /*num_layers=*/1, /*max_blocks=*/64,
                                                        kKVBlockSize, /*num_kv_heads=*/4, /*head_dim=*/64);
        ASSERT_TRUE(sr.has_value());

        auto mr = mgr_.init(std::move(*sr), 64, kKVBlockSize);
        ASSERT_TRUE(mr.has_value());
    }

    std::unique_ptr<DefaultBackend> backend_;
    KVCacheManager mgr_;
};

TEST_F(KVCacheManagerTest, InitState) {
    EXPECT_EQ(mgr_.num_free_blocks(), 64);
    EXPECT_EQ(mgr_.max_blocks(), 64);
    EXPECT_EQ(mgr_.block_size(), kKVBlockSize);
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

TEST_F(KVCacheManagerTest, PrepareBlocksNoPrefixHit) {
    std::vector<int32_t> tokens(32, 1);  // 2 blocks
    auto pr = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr.has_value());
    EXPECT_EQ(pr->block_table.size(), 2);
    EXPECT_EQ(pr->prefix_hit_blocks, 0);
    EXPECT_EQ(pr->block_table.shared_count(), 0);
    EXPECT_EQ(mgr_.num_free_blocks(), 62);
}

TEST_F(KVCacheManagerTest, PrepareBlocksWithPrefixHit) {
    // Request 1: 32 tokens, 2 blocks.
    std::vector<int32_t> tokens(32, 1);
    auto pr1 = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr1.has_value());

    // Cache them.
    auto cf = mgr_.cache_full_blocks(pr1->block_table, tokens, 32);
    ASSERT_TRUE(cf.has_value());

    // Release → blocks become CACHED_IDLE (in LRU, not free).
    mgr_.release_blocks(pr1->block_table);
    int free_after_release = mgr_.num_free_blocks();
    auto stats = mgr_.stats();
    EXPECT_GT(stats.block_cached_idle, 0);

    // Request 2: same tokens → should get prefix hit.
    auto pr2 = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr2.has_value());
    EXPECT_EQ(pr2->block_table.size(), 2);
    EXPECT_EQ(pr2->prefix_hit_blocks, 2);
    EXPECT_EQ(pr2->block_table.shared_count(), 2);
    // No new blocks allocated; all were from prefix cache.
    EXPECT_EQ(mgr_.num_free_blocks(), free_after_release);
}

TEST_F(KVCacheManagerTest, LookupPrefixCacheDoesNotAllocateSuffixBlocks) {
    std::vector<int32_t> cached_prefix(16, 7);
    auto pr1 = mgr_.prepare_blocks(cached_prefix);
    ASSERT_TRUE(pr1.has_value());
    ASSERT_TRUE(mgr_.cache_full_blocks(pr1->block_table, cached_prefix, 16).has_value());
    ASSERT_TRUE(mgr_.release_blocks(pr1->block_table).has_value());

    int free_before = mgr_.num_free_blocks();
    std::vector<int32_t> longer_tokens(32, 7);
    auto lookup = mgr_.lookup_prefix_cache(longer_tokens);
    ASSERT_TRUE(lookup.has_value());

    EXPECT_EQ(lookup->prefix_hit_blocks, 1);
    EXPECT_EQ(lookup->block_table.size(), 1);
    EXPECT_EQ(lookup->block_table.shared_count(), 1);
    EXPECT_EQ(mgr_.num_free_blocks(), free_before);

    ASSERT_TRUE(mgr_.release_blocks(lookup->block_table).has_value());
}

TEST_F(KVCacheManagerTest, PrefixHitRefCountBumped) {
    std::vector<int32_t> tokens(16, 7);  // 1 block
    auto pr1 = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr1.has_value());
    mgr_.cache_full_blocks(pr1->block_table, tokens, 16);
    mgr_.release_blocks(pr1->block_table);

    // Second request: hit.
    auto pr2 = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr2.has_value());
    EXPECT_EQ(pr2->prefix_hit_blocks, 1);

    // Block should be ACTIVE_CACHED (ref_count=1, not in LRU).
    auto s = mgr_.stats();
    EXPECT_EQ(s.block_cached_idle, 0);

    // Release second request → back to CACHED_IDLE.
    mgr_.release_blocks(pr2->block_table);
    EXPECT_GT(mgr_.stats().block_cached_idle, 0);
}

TEST_F(KVCacheManagerTest, LruEvictionWhenFreeListEmpty) {
    // Fill prefix cache with all 64 blocks using distinct tokens per request.
    for (int i = 0; i < 64; ++i) {
        std::vector<int32_t> tokens(16, i + 1);
        auto pr = mgr_.prepare_blocks(tokens);
        ASSERT_TRUE(pr.has_value());
        mgr_.cache_full_blocks(pr->block_table, tokens, 16);
        mgr_.release_blocks(pr->block_table);
    }
    EXPECT_EQ(mgr_.num_free_blocks(), 0);
    EXPECT_EQ(mgr_.stats().block_cached_idle, 64);

    // Next allocate should evict from LRU.
    auto alloc = mgr_.allocate_blocks(1);
    ASSERT_TRUE(alloc.has_value());
    auto s = mgr_.stats();
    EXPECT_EQ(s.block_cached_idle, 63);
    EXPECT_GT(s.prefix.evictions, 0);
}

TEST_F(KVCacheManagerTest, CacheFullBlocksRejectsUncommitted) {
    // 32 tokens = 2 blocks. Only commit 16 (1 block).
    std::vector<int32_t> tokens(32, 1);
    auto pr = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr.has_value());

    // Cache only first 16 tokens (1 block committed).
    auto cf = mgr_.cache_full_blocks(pr->block_table, tokens, 16);
    ASSERT_TRUE(cf.has_value());

    // Only 1 block should be cached (the committed one).
    mgr_.release_blocks(pr->block_table);
    auto s = mgr_.stats();
    EXPECT_EQ(s.block_cached_idle, 1);  // first block → LRU
    EXPECT_EQ(mgr_.num_free_blocks(), 63);  // second block → FREE
}

TEST_F(KVCacheManagerTest, StatsBlockAccounting) {
    auto s = mgr_.stats();
    EXPECT_EQ(s.block_total, 64);
    EXPECT_EQ(s.block_size, kKVBlockSize);
    EXPECT_EQ(s.block_free, 64);
    EXPECT_EQ(s.block_active, 0);
    EXPECT_EQ(s.block_cached_idle, 0);

    auto alloc = mgr_.allocate_blocks(2);
    ASSERT_TRUE(alloc.has_value());
    s = mgr_.stats();
    EXPECT_EQ(s.block_free, 62);
    EXPECT_EQ(s.block_active, 2);
}

TEST_F(KVCacheManagerTest, PrefixStatsReflectsCache) {
    auto ps = mgr_.prefix_stats();
    EXPECT_EQ(ps.cached_blocks, 0);

    std::vector<int32_t> tokens(16, 42);
    auto pr = mgr_.prepare_blocks(tokens);
    ASSERT_TRUE(pr.has_value());
    mgr_.cache_full_blocks(pr->block_table, tokens, 16);

    ps = mgr_.prefix_stats();
    EXPECT_EQ(ps.cached_blocks, 1);
    EXPECT_GT(ps.lookup_hits + ps.lookup_misses, 0);  // chain_hashes → lookup called
}

TEST_F(KVCacheManagerTest, ReleaseUncachedGoesToFree) {
    auto alloc = mgr_.allocate_blocks(1);
    ASSERT_TRUE(alloc.has_value());
    EXPECT_EQ(mgr_.num_free_blocks(), 63);

    mgr_.release_blocks(*alloc);
    EXPECT_EQ(mgr_.num_free_blocks(), 64);
    EXPECT_EQ(mgr_.stats().block_cached_idle, 0);
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
