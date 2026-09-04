#include <vector>

#include <gtest/gtest.h>

#include "block/block_pool.h"

namespace ccinfer {
namespace {

void publish_blocks(BlockPool& pool, const std::vector<int32_t>& tokens, const BlockTable& table,
                    int count) {
    uint64_t parent_hash = 0;
    for (int i = 0; i < count; ++i) {
        const auto begin = tokens.begin() + i * pool.block_size();
        const std::vector<int32_t> block_tokens(begin, begin + pool.block_size());
        parent_hash = pool.publish_full_block(parent_hash, block_tokens, table[i]);
    }
}

TEST(BlockPoolTest, InitState) {
    BlockPool pool(64, kKVBlockSize);

    EXPECT_EQ(pool.num_free_blocks(), 64);
    EXPECT_EQ(pool.max_blocks(), 64);
    EXPECT_EQ(pool.block_size(), kKVBlockSize);
}

TEST(BlockPoolTest, AllocateAndRelease) {
    BlockPool pool(64, kKVBlockSize);

    auto blocks = pool.allocate_blocks(2);
    ASSERT_TRUE(blocks);
    EXPECT_EQ(pool.num_free_blocks(), 62);

    pool.release_blocks(*blocks);
    EXPECT_EQ(pool.num_free_blocks(), 64);
}

TEST(BlockPoolTest, Exhaustion) {
    BlockPool pool(64, kKVBlockSize);
    for (int i = 0; i < 64; ++i) ASSERT_TRUE(pool.allocate_blocks(1));

    auto result = pool.allocate_blocks(1);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::KVBlockExhausted);
}

TEST(BlockPoolTest, LookupWithoutPrefixHitAllocatesOnlyMissingBlocks) {
    BlockPool pool(64, kKVBlockSize);
    const std::vector<int32_t> tokens(32, 1);

    auto lookup = pool.lookup_prefix_cache(tokens);
    ASSERT_EQ(lookup.prefix_hit_blocks, 0);
    auto blocks = pool.allocate_blocks(2 - lookup.block_table.size());
    ASSERT_TRUE(blocks);
    for (int i = 0; i < blocks->size(); ++i) lookup.block_table.push_back((*blocks)[i]);

    EXPECT_EQ(lookup.block_table.size(), 2);
    EXPECT_EQ(lookup.block_table.shared_count(), 0);
    EXPECT_EQ(pool.num_free_blocks(), 62);
}

TEST(BlockPoolTest, LookupReturnsPublishedPrefix) {
    BlockPool pool(64, kKVBlockSize);
    const std::vector<int32_t> tokens(32, 1);

    auto first = pool.lookup_prefix_cache(tokens);
    auto first_blocks = pool.allocate_blocks(2);
    ASSERT_TRUE(first_blocks);
    for (int i = 0; i < first_blocks->size(); ++i) first.block_table.push_back((*first_blocks)[i]);
    publish_blocks(pool, tokens, first.block_table, 2);
    pool.release_blocks(first.block_table);

    const int free_after_release = pool.num_free_blocks();
    auto second = pool.lookup_prefix_cache(tokens);
    EXPECT_EQ(second.prefix_hit_blocks, 2);
    EXPECT_EQ(second.block_table.size(), 2);
    EXPECT_EQ(second.block_table.shared_count(), 2);
    EXPECT_EQ(pool.num_free_blocks(), free_after_release);
    pool.release_blocks(second.block_table);
}

TEST(BlockPoolTest, PrefixLookupDoesNotAllocateSuffixBlocks) {
    BlockPool pool(64, kKVBlockSize);
    const std::vector<int32_t> prefix(16, 7);
    auto first = pool.lookup_prefix_cache(prefix);
    auto first_block = pool.allocate_blocks(1);
    ASSERT_TRUE(first_block);
    first.block_table.push_back((*first_block)[0]);
    publish_blocks(pool, prefix, first.block_table, 1);
    pool.release_blocks(first.block_table);

    const int free_before = pool.num_free_blocks();
    auto lookup = pool.lookup_prefix_cache(std::vector<int32_t>(32, 7));
    EXPECT_EQ(lookup.prefix_hit_blocks, 1);
    EXPECT_EQ(lookup.block_table.size(), 1);
    EXPECT_EQ(pool.num_free_blocks(), free_before);
    pool.release_blocks(lookup.block_table);
}

TEST(BlockPoolTest, PrefixHitRetainsBlockUntilRelease) {
    BlockPool pool(64, kKVBlockSize);
    const std::vector<int32_t> tokens(16, 7);
    auto first = pool.lookup_prefix_cache(tokens);
    auto first_block = pool.allocate_blocks(1);
    ASSERT_TRUE(first_block);
    first.block_table.push_back((*first_block)[0]);
    publish_blocks(pool, tokens, first.block_table, 1);
    pool.release_blocks(first.block_table);

    auto second = pool.lookup_prefix_cache(tokens);
    ASSERT_EQ(second.prefix_hit_blocks, 1);
    EXPECT_EQ(pool.stats().block_cached_idle, 0);
    pool.release_blocks(second.block_table);
    EXPECT_EQ(pool.stats().block_cached_idle, 1);
}

TEST(BlockPoolTest, LruEvictionReclaimsCachedBlock) {
    BlockPool pool(64, kKVBlockSize);
    for (int i = 0; i < 64; ++i) {
        const std::vector<int32_t> tokens(kKVBlockSize, i + 1);
        auto lookup = pool.lookup_prefix_cache(tokens);
        auto blocks = pool.allocate_blocks(1 - lookup.block_table.size());
        ASSERT_TRUE(blocks);
        for (int j = 0; j < blocks->size(); ++j) lookup.block_table.push_back((*blocks)[j]);
        publish_blocks(pool, tokens, lookup.block_table, 1);
        pool.release_blocks(lookup.block_table);
    }

    EXPECT_EQ(pool.num_free_blocks(), 0);
    EXPECT_EQ(pool.stats().block_cached_idle, 64);
    auto blocks = pool.allocate_blocks(1);
    ASSERT_TRUE(blocks);
    EXPECT_EQ(pool.stats().block_cached_idle, 63);
    EXPECT_GT(pool.prefix_stats().evictions, 0);
}

TEST(BlockPoolTest, UnpublishedSuffixReturnsToFreeList) {
    BlockPool pool(64, kKVBlockSize);
    const std::vector<int32_t> tokens(32, 1);
    auto lookup = pool.lookup_prefix_cache(tokens);
    auto blocks = pool.allocate_blocks(2);
    ASSERT_TRUE(blocks);
    for (int i = 0; i < blocks->size(); ++i) lookup.block_table.push_back((*blocks)[i]);
    publish_blocks(pool, tokens, lookup.block_table, 1);
    pool.release_blocks(lookup.block_table);

    EXPECT_EQ(pool.stats().block_cached_idle, 1);
    EXPECT_EQ(pool.num_free_blocks(), 63);
}

TEST(BlockPoolTest, StatsTrackBlocksAndPrefixEntries) {
    BlockPool pool(64, kKVBlockSize);
    auto stats = pool.stats();
    EXPECT_EQ(stats.block_total, 64);
    EXPECT_EQ(stats.block_size, kKVBlockSize);
    EXPECT_EQ(stats.block_free, 64);
    EXPECT_EQ(stats.block_active, 0);
    EXPECT_EQ(stats.block_cached_idle, 0);

    auto blocks = pool.allocate_blocks(2);
    ASSERT_TRUE(blocks);
    stats = pool.stats();
    EXPECT_EQ(stats.block_free, 62);
    EXPECT_EQ(stats.block_active, 2);

    const std::vector<int32_t> tokens(kKVBlockSize, 42);
    publish_blocks(pool, tokens, *blocks, 1);
    EXPECT_EQ(pool.prefix_stats().cached_blocks, 1);
}

TEST(BlockPoolTest, DuplicatePublicationKeepsRequestBlockId) {
    BlockPool pool(4, 4);
    const std::vector<int32_t> tokens(4, 9);

    auto first = pool.allocate_blocks(1);
    ASSERT_TRUE(first);
    const int32_t first_id = (*first)[0];
    pool.publish_full_block(0, tokens, first_id);
    pool.release_blocks(*first);

    auto lookup = pool.lookup_prefix_cache(tokens);
    ASSERT_EQ(lookup.prefix_hit_blocks, 1);
    EXPECT_EQ(lookup.block_table[0], first_id);

    auto duplicate = pool.allocate_blocks(1);
    ASSERT_TRUE(duplicate);
    const int32_t duplicate_id = (*duplicate)[0];
    pool.publish_full_block(0, tokens, duplicate_id);
    EXPECT_EQ((*duplicate)[0], duplicate_id);
    EXPECT_EQ(pool.prefix_stats().cached_blocks, 2u);

    pool.release_blocks(lookup.block_table);
    pool.release_blocks(*duplicate);
    EXPECT_EQ(pool.prefix_stats().cached_blocks, 2u);
}

TEST(BlockPoolTest, EvictionKeepsNextDuplicateCandidate) {
    BlockPool pool(2, 4);
    const std::vector<int32_t> tokens(4, 11);

    auto canonical = pool.allocate_blocks(1);
    ASSERT_TRUE(canonical);
    const int32_t canonical_id = (*canonical)[0];
    pool.publish_full_block(0, tokens, canonical_id);
    pool.release_blocks(*canonical);

    auto lookup = pool.lookup_prefix_cache(tokens);
    auto duplicate = pool.allocate_blocks(1);
    ASSERT_TRUE(duplicate);
    const int32_t duplicate_id = (*duplicate)[0];
    pool.publish_full_block(0, tokens, duplicate_id);
    pool.release_blocks(lookup.block_table);

    auto replacement = pool.allocate_blocks(1);
    ASSERT_TRUE(replacement);
    EXPECT_EQ((*replacement)[0], canonical_id);
    auto after_eviction = pool.lookup_prefix_cache(tokens);
    ASSERT_EQ(after_eviction.prefix_hit_blocks, 1);
    EXPECT_EQ(after_eviction.block_table[0], duplicate_id);

    pool.release_blocks(after_eviction.block_table);
    pool.release_blocks(*replacement);
    pool.release_blocks(*duplicate);
}

}  // namespace
}  // namespace ccinfer
