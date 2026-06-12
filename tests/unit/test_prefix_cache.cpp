#include <gtest/gtest.h>

#include "cache/prefix_cache.h"

namespace ccinfer {
namespace {

TEST(PrefixCacheTest, ChainHashesDeterministic) {
    std::vector<int32_t> tokens(16, 42);  // 1 full block of 42s
    auto h1 = PrefixCache::chain_hashes(tokens, 16, /*namespace_salt=*/0);
    auto h2 = PrefixCache::chain_hashes(tokens, 16, /*namespace_salt=*/0);

    ASSERT_EQ(h1.size(), 1);
    ASSERT_EQ(h2.size(), 1);
    EXPECT_EQ(h1[0], h2[0]);
}

TEST(PrefixCacheTest, ChainHashesTwoFullBlocks) {
    std::vector<int32_t> tokens(32, 1);  // 2 full blocks
    for (int i = 16; i < 32; ++i) tokens[i] = 2;

    auto hashes = PrefixCache::chain_hashes(tokens, 16, /*namespace_salt=*/0);
    ASSERT_EQ(hashes.size(), 2);
    EXPECT_NE(hashes[0], hashes[1]);
}

TEST(PrefixCacheTest, ChainHashesPartialTailOmitted) {
    std::vector<int32_t> tokens(20, 1);  // 1 full block + 4 remainder
    auto hashes = PrefixCache::chain_hashes(tokens, 16);
    EXPECT_EQ(hashes.size(), 1);
}

TEST(PrefixCacheTest, ChainHashesEmptyTokens) {
    auto hashes = PrefixCache::chain_hashes({}, 16);
    EXPECT_TRUE(hashes.empty());
}

TEST(PrefixCacheTest, ChainHashesSaltDifferentiates) {
    std::vector<int32_t> tokens(16, 7);
    auto h0 = PrefixCache::chain_hashes(tokens, 16, /*namespace_salt=*/0);
    auto h1 = PrefixCache::chain_hashes(tokens, 16, /*namespace_salt=*/1);
    ASSERT_EQ(h0.size(), 1);
    ASSERT_EQ(h1.size(), 1);
    EXPECT_NE(h0[0], h1[0]);
}

TEST(PrefixCacheTest, ChainHashesTokenCountOverload) {
    std::vector<int32_t> tokens(32, 1);
    for (int i = 16; i < 32; ++i) tokens[i] = 2;

    // Only hash first 16 tokens.
    auto hashes = PrefixCache::chain_hashes(tokens, 16, 16, 0);
    ASSERT_EQ(hashes.size(), 1);

    // Verify it matches the single-block hash.
    std::vector<int32_t> first16(tokens.begin(), tokens.begin() + 16);
    auto ref = PrefixCache::chain_hashes(first16, 16, 0);
    EXPECT_EQ(hashes[0], ref[0]);
}

TEST(PrefixCacheTest, InsertLookupRoundTrip) {
    PrefixCache cache;
    auto r = cache.insert(0xABCD, 3);
    ASSERT_TRUE(r.has_value());

    auto opt = cache.lookup(0xABCD);
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(*opt, 3);
}

TEST(PrefixCacheTest, LookupMissReturnsNullopt) {
    PrefixCache cache;
    EXPECT_FALSE(cache.lookup(0xDEAD).has_value());
}

TEST(PrefixCacheTest, InsertSameBlockIdempotent) {
    PrefixCache cache;
    ASSERT_TRUE(cache.insert(0xAAAA, 7).has_value());
    ASSERT_TRUE(cache.insert(0xAAAA, 7).has_value());  // idempotent
    EXPECT_EQ(cache.size(), 1);
}

TEST(PrefixCacheTest, InsertDifferentHashSameBlockCollision) {
    PrefixCache cache;
    ASSERT_TRUE(cache.insert(0xAAAA, 7).has_value());
    auto r = cache.insert(0xBBBB, 7);  // block 7 already maps to 0xAAAA
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::KVBlockHashCollision);
}

TEST(PrefixCacheTest, InsertSameHashDifferentBlockCollision) {
    PrefixCache cache;
    ASSERT_TRUE(cache.insert(0xAAAA, 7).has_value());
    auto r = cache.insert(0xAAAA, 8);  // hash 0xAAAA already maps to block 7
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::KVBlockHashCollision);
}

TEST(PrefixCacheTest, RemoveByHash) {
    PrefixCache cache;
    cache.insert(0xABCD, 5);
    cache.remove(0xABCD);
    EXPECT_FALSE(cache.lookup(0xABCD).has_value());
    EXPECT_EQ(cache.size(), 0);
}

TEST(PrefixCacheTest, RemoveByBlock) {
    PrefixCache cache;
    cache.insert(0xABCD, 5);
    cache.remove_by_block(5);
    EXPECT_FALSE(cache.lookup(0xABCD).has_value());
    EXPECT_EQ(cache.size(), 0);
}

TEST(PrefixCacheTest, LookupAutoStats) {
    PrefixCache cache;
    cache.insert(0xAAAA, 1);
    cache.insert(0xBBBB, 2);

    cache.lookup(0xAAAA);  // hit
    cache.lookup(0xBBBB);  // hit
    cache.lookup(0xCCCC);  // miss

    auto s = cache.stats();
    EXPECT_EQ(s.lookup_hits, 2);
    EXPECT_EQ(s.lookup_misses, 1);
}

TEST(PrefixCacheTest, EvictionStats) {
    PrefixCache cache;
    cache.record_eviction();
    cache.record_eviction();
    EXPECT_EQ(cache.stats().evictions, 2);
}

TEST(PrefixCacheTest, CachedBlocksCount) {
    PrefixCache cache;
    EXPECT_EQ(cache.stats().cached_blocks, 0);
    cache.insert(0x1111, 1);
    cache.insert(0x2222, 2);
    cache.insert(0x3333, 3);
    EXPECT_EQ(cache.stats().cached_blocks, 3);
}

}  // namespace
}  // namespace ccinfer
