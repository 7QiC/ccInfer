#include <vector>

#include <gtest/gtest.h>

#include "block/block_pool.h"
#include "cache/block_cache.h"
#include "cache/prefix_cache.h"

namespace ccinfer {
namespace {

void publish_blocks(BlockPool& pool, BlockCache& cache, const std::vector<int32_t>& tokens,
                    const BlockTable& table) {
    uint64_t parent_hash = 0;
    for (int i = 0; i < table.size(); ++i) {
        const auto begin = tokens.begin() + i * pool.block_size();
        const std::vector<int32_t> block_tokens(begin, begin + pool.block_size());
        parent_hash = cache.publish_full_block(parent_hash, block_tokens, table[i]);
    }
}

void release_request_blocks(BlockPool& pool, BlockCache& cache, const BlockTable& table) {
    for (int i = 0; i < table.size(); ++i) {
        const BlockId id = table[i];
        if (pool.release_request(id) != 0) continue;
        if (cache.contains(id)) {
            cache.mark_idle(id);
        } else {
            pool.recycle(id);
        }
    }
}

Result<BlockTable> allocate_with_eviction(BlockPool& pool, BlockCache& cache, int num_blocks) {
    while (pool.num_free_blocks() < num_blocks) {
        if (!cache.evict_one()) break;
    }
    return pool.allocate_blocks(num_blocks);
}

TEST(BlockCacheTest, PublishReleaseMarksIdle) {
    BlockPool pool(4, kKVBlockSize);
    BlockCache cache(pool);
    EXPECT_EQ(cache.num_cached_blocks(), 0u);

    auto blocks = pool.allocate_blocks(1);
    ASSERT_TRUE(blocks);
    const std::vector<int32_t> tokens(kKVBlockSize, 1);
    publish_blocks(pool, cache, tokens, *blocks);
    EXPECT_EQ(cache.num_cached_blocks(), 1u);
    EXPECT_EQ(cache.num_idle_blocks(), 0u);
    EXPECT_EQ(pool.num_free_blocks(), 3);

    release_request_blocks(pool, cache, *blocks);
    EXPECT_EQ(cache.num_idle_blocks(), 1u);
    EXPECT_EQ(pool.num_free_blocks(), 3);
    EXPECT_EQ(pool.ref_count((*blocks)[0]), 0);
    EXPECT_TRUE(cache.is_idle((*blocks)[0]));
}

TEST(BlockCacheTest, LookupHitPromotesIdleToActive) {
    BlockPool pool(4, kKVBlockSize);
    BlockCache cache(pool);
    const std::vector<int32_t> tokens(kKVBlockSize, 7);

    auto first = pool.allocate_blocks(1);
    ASSERT_TRUE(first);
    publish_blocks(pool, cache, tokens, *first);
    release_request_blocks(pool, cache, *first);

    auto lookup = cache.lookup_prefix_cache(tokens);
    ASSERT_EQ(lookup.prefix_hit_blocks, 1);
    ASSERT_EQ(lookup.block_table.size(), 1);
    EXPECT_EQ(lookup.block_table[0], (*first)[0]);
    EXPECT_EQ(cache.num_idle_blocks(), 0u);
    EXPECT_EQ(pool.ref_count((*first)[0]), 1);

    release_request_blocks(pool, cache, lookup.block_table);
    EXPECT_EQ(cache.num_idle_blocks(), 1u);
}

TEST(BlockCacheTest, LruEvictionRecyclesIdleBlock) {
    BlockPool pool(4, kKVBlockSize);
    BlockCache cache(pool);
    std::vector<BlockTable> tables;
    for (int i = 0; i < 4; ++i) {
        auto blocks = pool.allocate_blocks(1);
        ASSERT_TRUE(blocks);
        const std::vector<int32_t> tokens(kKVBlockSize, i + 1);
        publish_blocks(pool, cache, tokens, *blocks);
        release_request_blocks(pool, cache, *blocks);
        tables.push_back(std::move(*blocks));
    }
    EXPECT_EQ(pool.num_free_blocks(), 0);
    EXPECT_EQ(cache.num_idle_blocks(), 4u);

    auto blocks = allocate_with_eviction(pool, cache, 1);
    ASSERT_TRUE(blocks);
    EXPECT_EQ(cache.num_idle_blocks(), 3u);
    EXPECT_GT(cache.stats().evictions, 0u);
}

TEST(BlockCacheTest, LookupWithoutHitDoesNotAllocate) {
    BlockPool pool(4, kKVBlockSize);
    BlockCache cache(pool);
    const std::vector<int32_t> tokens(kKVBlockSize, 3);
    auto lookup = cache.lookup_prefix_cache(tokens);
    EXPECT_EQ(lookup.prefix_hit_blocks, 0);
    EXPECT_TRUE(lookup.block_table.empty());
    EXPECT_EQ(pool.num_free_blocks(), 4);
}

TEST(BlockCacheTest, PrefixLookupDoesNotAllocateSuffixBlocks) {
    BlockPool pool(4, kKVBlockSize);
    BlockCache cache(pool);
    const std::vector<int32_t> prefix(kKVBlockSize, 7);
    auto blocks = pool.allocate_blocks(1);
    ASSERT_TRUE(blocks);
    publish_blocks(pool, cache, prefix, *blocks);
    release_request_blocks(pool, cache, *blocks);

    const int free_before = pool.num_free_blocks();
    auto lookup = cache.lookup_prefix_cache(std::vector<int32_t>(2 * kKVBlockSize, 7));
    EXPECT_EQ(lookup.prefix_hit_blocks, 1);
    EXPECT_EQ(lookup.block_table.size(), 1);
    EXPECT_EQ(pool.num_free_blocks(), free_before);
    release_request_blocks(pool, cache, lookup.block_table);
}

TEST(BlockCacheTest, DuplicatePrefixKeepsEvictionCandidate) {
    BlockPool pool(2, kKVBlockSize);
    BlockCache cache(pool);
    const std::vector<int32_t> tokens(kKVBlockSize, 11);

    auto canonical = pool.allocate_blocks(1);
    ASSERT_TRUE(canonical);
    publish_blocks(pool, cache, tokens, *canonical);
    release_request_blocks(pool, cache, *canonical);

    auto duplicate = pool.allocate_blocks(1);
    ASSERT_TRUE(duplicate);
    publish_blocks(pool, cache, tokens, *duplicate);
    release_request_blocks(pool, cache, *duplicate);
    EXPECT_EQ(cache.num_cached_blocks(), 2u);

    auto replacement = allocate_with_eviction(pool, cache, 1);
    ASSERT_TRUE(replacement);
    auto after = cache.lookup_prefix_cache(tokens);
    ASSERT_EQ(after.prefix_hit_blocks, 1);
    EXPECT_EQ(after.block_table[0], (*duplicate)[0]);

    release_request_blocks(pool, cache, after.block_table);
    release_request_blocks(pool, cache, *replacement);
}

}  // namespace
}  // namespace ccinfer
