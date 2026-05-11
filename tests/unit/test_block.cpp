#include <gtest/gtest.h>

#include "engine/cache/block.h"

namespace ccinfer {
namespace engine {
namespace {

TEST(BlockTest, InitialState) {
    Block b;
    EXPECT_EQ(b.block_id, -1);
    EXPECT_EQ(b.ref_count, 0);
    EXPECT_EQ(b.block_hash, 0);
    EXPECT_FALSE(b.is_free());
    EXPECT_FALSE(b.is_cached());
}

TEST(BlockTest, Flags) {
    Block b;
    b.flags = static_cast<uint32_t>(BlockFlags::kInFreeList);
    EXPECT_TRUE(b.is_free());
    EXPECT_FALSE(b.is_cached());

    b.flags = static_cast<uint32_t>(BlockFlags::kInFreeList) | static_cast<uint32_t>(BlockFlags::kCached);
    EXPECT_TRUE(b.is_free());
    EXPECT_TRUE(b.is_cached());
}

TEST(BlockTest, FreeListPushPop) {
    FreeList fl;
    Block blocks[3];
    for (int i = 0; i < 3; ++i) blocks[i].block_id = i;

    fl.push_back(blocks[0]);
    fl.push_back(blocks[1]);
    fl.push_back(blocks[2]);
    EXPECT_EQ(static_cast<int>(fl.size()), 3);

    auto& front = fl.front();
    fl.pop_front();
    EXPECT_EQ(front.block_id, 0);
    EXPECT_EQ(static_cast<int>(fl.size()), 2);

    fl.pop_front();
    fl.pop_front();
}

TEST(BlockTableTest, Empty) {
    BlockTable bt;
    EXPECT_EQ(bt.size(), 0);
    EXPECT_EQ(bt.token_capacity(kKVBlockSize), 0);
}

TEST(BlockTableTest, PushBackAndIndex) {
    BlockTable bt;
    bt.push_back(7);
    bt.push_back(3);
    bt.push_back(42);

    EXPECT_EQ(bt.size(), 3);
    EXPECT_EQ(bt[0], 7);
    EXPECT_EQ(bt[1], 3);
    EXPECT_EQ(bt[2], 42);
    EXPECT_EQ(bt.token_capacity(kKVBlockSize), 3 * kKVBlockSize);
}

TEST(BlockTableTest, SharedCount) {
    BlockTable bt;
    bt.push_back(10);
    bt.push_back(20);
    bt.set_shared_count(1);

    EXPECT_EQ(bt.shared_count(), 1);
    EXPECT_EQ(bt[0], 10);
}

TEST(BlockTableTest, DataPointer) {
    BlockTable bt;
    bt.push_back(5);
    bt.push_back(8);

    const int32_t* ptr = bt.data();
    EXPECT_EQ(ptr[0], 5);
    EXPECT_EQ(ptr[1], 8);
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
