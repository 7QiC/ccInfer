#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "backend/backend.h"
#include "block/block.h"
#include "worker/worker.h"

namespace ccinfer {

namespace {

class WorkerBatchTranslatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto backend = Backend::create(0);
        if (!backend) GTEST_SKIP() << "CUDA unavailable";
        backend_ = std::move(*backend);
    }

    std::unique_ptr<Backend> backend_;
};

TEST_F(WorkerBatchTranslatorTest, TranslatesSchedulerPreparedPrefill) {
    BlockTable blocks;
    blocks.push_back(0);

    ScheduledBatch batch;
    batch.batch_id = 1;
    batch.items.push_back(
        PrefillChunk{1, TokenSpan{0, 16}, 0, false, std::vector<int32_t>(16, 1), blocks});
    auto result = Worker::BatchTranslator(*backend_, kKVBlockSize).translate(batch);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->num_tokens, 16);
    EXPECT_EQ(result->batch_size, 1);
    EXPECT_EQ(result->max_blocks_per_req, 1);
    EXPECT_EQ(result->mode, ForwardMode::Prefill);
}

TEST_F(WorkerBatchTranslatorTest, TranslatesDecodeWithoutAllocating) {
    BlockTable blocks;
    blocks.push_back(0);
    blocks.push_back(1);

    ScheduledBatch batch;
    batch.batch_id = 2;
    batch.items.push_back(DecodeOneToken{1, 42, 16, true, false, blocks});
    auto result = Worker::BatchTranslator(*backend_, kKVBlockSize).translate(batch);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->num_tokens, 1);
    EXPECT_EQ(result->context_lens.valid(), true);
    EXPECT_EQ(result->mode, ForwardMode::Decode);
}

}  // namespace
}  // namespace ccinfer
