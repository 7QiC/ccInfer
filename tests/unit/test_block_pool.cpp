#include <vector>

#include <gtest/gtest.h>

#include "block/block_pool.h"

namespace ccinfer {
namespace {

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
    EXPECT_EQ(pool.ref_count((*blocks)[0]), 1);

    pool.release_blocks(*blocks);
    EXPECT_EQ(pool.ref_count((*blocks)[0]), 0);
    // release_request must not recycle automatically.
    EXPECT_EQ(pool.num_free_blocks(), 62);
    pool.recycle((*blocks)[0]);
    pool.recycle((*blocks)[1]);
    EXPECT_EQ(pool.num_free_blocks(), 64);
}

TEST(BlockPoolTest, RetainAndReleaseRequestRefs) {
    BlockPool pool(4, kKVBlockSize);
    auto block = pool.allocate();
    ASSERT_TRUE(block);
    pool.retain_request(*block);
    EXPECT_EQ(pool.ref_count(*block), 2);
    EXPECT_EQ(pool.release_request(*block), 1);
    EXPECT_EQ(pool.release_request(*block), 0);
    EXPECT_EQ(pool.num_free_blocks(), 3);
    pool.recycle(*block);
    EXPECT_EQ(pool.num_free_blocks(), 4);
}

TEST(BlockPoolTest, Exhaustion) {
    BlockPool pool(64, kKVBlockSize);
    std::vector<BlockTable> allocations;
    allocations.reserve(64);
    for (int i = 0; i < 64; ++i) {
        auto block = pool.allocate();
        ASSERT_TRUE(block);
        BlockTable table;
        table.push_back(*block);
        allocations.push_back(std::move(table));
    }

    auto result = pool.allocate();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), ErrorCode::KVBlockExhausted);
}

TEST(BlockPoolTest, RecycleAfterReleaseReturnsToFreeList) {
    BlockPool pool(2, kKVBlockSize);
    auto first = pool.allocate();
    auto second = pool.allocate();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());

    ASSERT_EQ(pool.release_request(*first), 0);
    pool.recycle(*first);
    EXPECT_EQ(pool.num_free_blocks(), 1);

    ASSERT_EQ(pool.release_request(*second), 0);
    pool.recycle(*second);
    EXPECT_EQ(pool.num_free_blocks(), 2);
}

}  // namespace
}  // namespace ccinfer
