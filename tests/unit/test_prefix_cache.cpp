#include <vector>

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

}  // namespace
}  // namespace ccinfer
