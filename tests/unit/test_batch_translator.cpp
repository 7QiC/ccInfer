#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <memory>
#include <unordered_map>

#include "engine/cache/kv_cache_manager.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/common/backend_def.h"
#include "engine/common/types.h"
#include "engine/worker/batch_translator.h"

namespace ccinfer {
namespace engine {
namespace {

class BatchTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto b = CudaBackend::create(0);
        ASSERT_TRUE(b.has_value());
        backend_ = std::move(*b);

        auto sr = KVCacheStorage::create<__nv_bfloat16>(*backend_, /*num_layers=*/1, /*max_blocks=*/32,
                                                        kBlockSize, /*num_kv_heads=*/4, /*head_dim=*/64);
        ASSERT_TRUE(sr.has_value());

        auto r = kv_mgr_.init(std::move(*sr), 32, kBlockSize);
        ASSERT_TRUE(r.has_value());
        translator_ = std::make_unique<BatchTranslator>(*backend_, kv_mgr_, kBlockSize);

        // Create one sequence with a 20-token prompt.
        SequenceState seq;
        seq.seq_id = 1;
        seq.prompt_tokens = std::vector<int32_t>(20, 1);
        seq.max_context_len = 64;
        sequences_[1] = std::move(seq);
    }

    static constexpr int kBlockSize = 16;
    std::unique_ptr<DefaultBackend> backend_;
    KVCacheManager kv_mgr_;
    std::unique_ptr<BatchTranslator> translator_;
    std::unordered_map<SequenceId, SequenceState> sequences_;
};

TEST_F(BatchTranslatorTest, PrefillAllocatesBlocks) {
    ScheduledBatch batch;
    batch.batch_id = 1;
    batch.items.push_back(PrefillChunk{1, TokenSpan{0, 16}, std::nullopt});

    int free_before = kv_mgr_.num_free_blocks();

    auto result = translator_->translate(batch, sequences_);
    ASSERT_TRUE(result.has_value());

    // 16 tokens → 1 block allocated.
    EXPECT_LT(kv_mgr_.num_free_blocks(), free_before);

    const auto& pb = result->physical_batch;
    EXPECT_EQ(pb.num_tokens, 16);
    EXPECT_EQ(pb.batch_size, 1);
    EXPECT_EQ(pb.max_blocks_per_req, 1);

    // Commit: kv_written and prompt_processed advance.
    auto commit_r = translator_->commit(batch, sequences_, result->per_item);
    ASSERT_TRUE(commit_r.has_value());
    EXPECT_EQ(sequences_[1].kv_written, 16);
    EXPECT_EQ(sequences_[1].prompt_processed, 16);
    EXPECT_EQ(sequences_[1].block_table.size(), 1);
}

TEST_F(BatchTranslatorTest, RollbackRestoresFreeBlocks) {
    ScheduledBatch batch;
    batch.batch_id = 1;
    batch.items.push_back(PrefillChunk{1, TokenSpan{0, 16}, std::nullopt});

    int free_before = kv_mgr_.num_free_blocks();

    auto result = translator_->translate(batch, sequences_);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(kv_mgr_.num_free_blocks(), free_before);

    // Rollback: blocks back to free list, SequenceState unchanged.
    translator_->rollback(result->per_item);
    EXPECT_EQ(kv_mgr_.num_free_blocks(), free_before);
    EXPECT_EQ(sequences_[1].kv_written, 0);
    EXPECT_EQ(sequences_[1].block_table.size(), 0);
}

TEST_F(BatchTranslatorTest, DecodeNoNewBlock) {
    // Set up: prefill already done (8-token prompt), context fits in 1 block.
    sequences_[1].prompt_tokens = std::vector<int32_t>(8, 1);
    sequences_[1].kv_written = 8;
    sequences_[1].prompt_processed = 8;  // prefill complete
    auto alloc = kv_mgr_.allocate_blocks(1);  // 16 tokens → 1 block
    ASSERT_TRUE(alloc.has_value());
    for (int i = 0; i < alloc->size(); ++i) {
        sequences_[1].block_table.push_back((*alloc)[i]);
    }

    ScheduledBatch batch;
    batch.batch_id = 2;
    batch.items.push_back(DecodeOneToken{1, 42, std::nullopt});

    int free_before = kv_mgr_.num_free_blocks();

    auto result = translator_->translate(batch, sequences_);
    ASSERT_TRUE(result.has_value());

    // Decode should not allocate a new block (8 + 1 ≤ 16).
    EXPECT_EQ(kv_mgr_.num_free_blocks(), free_before);

    const auto& pb = result->physical_batch;
    EXPECT_EQ(pb.num_tokens, 1);
    EXPECT_EQ(pb.batch_size, 1);

    auto commit_r = translator_->commit(batch, sequences_, result->per_item);
    ASSERT_TRUE(commit_r.has_value());
    EXPECT_EQ(sequences_[1].kv_written, 9);
    EXPECT_EQ(sequences_[1].block_table.size(), 1);  // unchanged
}

TEST_F(BatchTranslatorTest, PrefillSpansMultipleBlocks) {
    ScheduledBatch batch;
    batch.batch_id = 1;
    batch.items.push_back(PrefillChunk{1, TokenSpan{0, 20}, std::nullopt});

    auto result = translator_->translate(batch, sequences_);
    ASSERT_TRUE(result.has_value());

    // 20 tokens → 2 blocks (ceil(20/16)).
    EXPECT_EQ(result->physical_batch.max_blocks_per_req, 2);
    EXPECT_EQ(result->physical_batch.num_tokens, 20);

    auto commit_r = translator_->commit(batch, sequences_, result->per_item);
    ASSERT_TRUE(commit_r.has_value());
    EXPECT_EQ(sequences_[1].kv_written, 20);
    EXPECT_EQ(sequences_[1].block_table.size(), 2);
}

TEST_F(BatchTranslatorTest, EmptyBatchReturnsError) {
    ScheduledBatch batch;
    batch.batch_id = 1;

    auto result = translator_->translate(batch, sequences_);
    EXPECT_FALSE(result.has_value());
}

TEST_F(BatchTranslatorTest, MissingSequenceReturnsError) {
    ScheduledBatch batch;
    batch.batch_id = 1;
    batch.items.push_back(DecodeOneToken{999, 42, std::nullopt});  // nonexistent

    auto result = translator_->translate(batch, sequences_);
    EXPECT_FALSE(result.has_value());
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
