#include <gtest/gtest.h>

#include <unistd.h>

#include <string>

#define private public
#include "server/scheduler/scheduler.h"
#undef private

#include "common/types.h"

namespace ccinfer {
namespace server {
namespace {

class SchedulerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto path = std::getenv("CCINFER_TEST_MODEL_PATH")
                        ? std::getenv("CCINFER_TEST_MODEL_PATH")
                        : "../models/qwen3-0.6B";
        std::string mp(path);
        if (access((mp + "/model.safetensors").c_str(), F_OK) != 0 ||
            access((mp + "/config.json").c_str(), F_OK) != 0) {
            suite_skip_ = true;
            return;
        }

        engine_ = std::make_unique<Engine>(io_);
        if (auto r = engine_->init(path); !r) {
            suite_skip_ = true;
            engine_.reset();
            return;
        }
    }

    static void TearDownTestSuite() {
        engine_.reset();
    }

    void SetUp() override {
        if (suite_skip_) GTEST_SKIP() << "Engine init failed (check CCINFER_TEST_MODEL_PATH)";
        scheduler_ = std::make_unique<Scheduler>(io_, *engine_);
    }

    void TearDown() override { scheduler_.reset(); }

    static boost::asio::io_context io_;
    static std::unique_ptr<Engine> engine_;
    static bool suite_skip_;

    std::unique_ptr<Scheduler> scheduler_;
};

boost::asio::io_context SchedulerTest::io_{};
std::unique_ptr<Engine> SchedulerTest::engine_;
bool SchedulerTest::suite_skip_ = false;

TEST_F(SchedulerTest, EmptyBatchWhenNoActiveSequences) {
    auto batch = scheduler_->build_scheduled_batch();
    EXPECT_TRUE(batch.items.empty());
}

TEST_F(SchedulerTest, SingleDecodeItem) {
    auto state = std::make_shared<SchedulerRequestState>();
    state->seq_id = 1;
    state->prefill_done = true;
    state->last_token = 7;
    state->prompt_tokens = {1};
    state->sampling.max_tokens = 10;

    scheduler_->active_[1] = state;
    scheduler_->active_order_.push_back(1);

    auto batch = scheduler_->build_scheduled_batch();
    ASSERT_EQ(batch.items.size(), 1u);
    auto* d = std::get_if<DecodeOneToken>(&batch.items[0]);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(d->seq_id, 1u);
    EXPECT_EQ(d->input_token, 7);
}

TEST_F(SchedulerTest, SkipsFinishedAndCancelled) {
    auto finished = std::make_shared<SchedulerRequestState>();
    finished->seq_id = 1; finished->prefill_done = true; finished->finished = true;

    auto cancelled = std::make_shared<SchedulerRequestState>();
    cancelled->seq_id = 2; cancelled->prefill_done = true; cancelled->cancelled = true;

    auto active = std::make_shared<SchedulerRequestState>();
    active->seq_id = 3; active->prefill_done = true;
    active->last_token = 5; active->prompt_tokens = {1};
    active->sampling.max_tokens = 10;

    scheduler_->active_[1] = finished;
    scheduler_->active_[2] = cancelled;
    scheduler_->active_[3] = active;
    scheduler_->active_order_ = {1, 2, 3};

    auto batch = scheduler_->build_scheduled_batch();
    ASSERT_EQ(batch.items.size(), 1u);
    EXPECT_EQ(std::get<DecodeOneToken>(batch.items[0]).seq_id, 3u);
}

TEST_F(SchedulerTest, MultipleDecodeItemsBatched) {
    auto mk = [](int id) {
        auto s = std::make_shared<SchedulerRequestState>();
        s->seq_id = id; s->prefill_done = true;
        s->last_token = id * 10; s->prompt_tokens = {1};
        s->sampling.max_tokens = 10;
        return s;
    };
    scheduler_->active_[1] = mk(1);
    scheduler_->active_[2] = mk(2);
    scheduler_->active_[3] = mk(3);
    scheduler_->active_order_ = {1, 2, 3};

    auto batch = scheduler_->build_scheduled_batch();
    ASSERT_EQ(batch.items.size(), 3u);
    for (size_t i = 0; i < 3; ++i)
        ASSERT_TRUE(std::holds_alternative<DecodeOneToken>(batch.items[i]));
}

TEST_F(SchedulerTest, SkipsIncompatibleSampling) {
    auto s1 = std::make_shared<SchedulerRequestState>();
    s1->seq_id = 1; s1->prefill_done = true;
    s1->last_token = 10; s1->prompt_tokens = {1};
    s1->sampling.max_tokens = 10;
    s1->sampling.temperature = 0.0f;

    auto s2 = std::make_shared<SchedulerRequestState>();
    s2->seq_id = 2; s2->prefill_done = true;
    s2->last_token = 20; s2->prompt_tokens = {1};
    s2->sampling.max_tokens = 10;
    s2->sampling.temperature = 0.5f;

    scheduler_->active_[1] = s1;
    scheduler_->active_[2] = s2;
    scheduler_->active_order_ = {1, 2};

    auto batch = scheduler_->build_scheduled_batch();
    ASSERT_EQ(batch.items.size(), 1u);
    EXPECT_EQ(std::get<DecodeOneToken>(batch.items[0]).seq_id, 1u);
}

TEST_F(SchedulerTest, SkipsMaxTokensReached) {
    auto state = std::make_shared<SchedulerRequestState>();
    state->seq_id = 1; state->prefill_done = true;
    state->last_token = 7;
    state->prompt_tokens = {1};
    state->tokens_generated = 10;
    state->sampling.max_tokens = 10;

    scheduler_->active_[1] = state;
    scheduler_->active_order_.push_back(1);

    auto batch = scheduler_->build_scheduled_batch();
    EXPECT_TRUE(batch.items.empty());
    EXPECT_TRUE(state->finished);
}

TEST_F(SchedulerTest, PrefillPriorityOverDecode) {
    auto prefill = std::make_shared<SchedulerRequestState>();
    prefill->seq_id = 1;
    prefill->prefill_done = false;
    prefill->prompt_tokens = {1, 2, 3, 4, 5};
    prefill->sampling.max_tokens = 10;

    auto decode = std::make_shared<SchedulerRequestState>();
    decode->seq_id = 2;
    decode->prefill_done = true;
    decode->last_token = 99;
    decode->prompt_tokens = {1};
    decode->sampling.max_tokens = 10;

    scheduler_->active_[1] = prefill;
    scheduler_->active_[2] = decode;
    scheduler_->active_order_ = {1, 2};

    auto batch = scheduler_->build_scheduled_batch();
    ASSERT_EQ(batch.items.size(), 1u);
    EXPECT_TRUE(std::holds_alternative<PrefillChunk>(batch.items[0]));
}

TEST_F(SchedulerTest, RoundRobinReordersAfterDecodeBatch) {
    auto mk = [](int id) {
        auto s = std::make_shared<SchedulerRequestState>();
        s->seq_id = id; s->prefill_done = true;
        s->last_token = id * 10; s->prompt_tokens = {1};
        s->sampling.max_tokens = 10;
        return s;
    };
    scheduler_->active_[1] = mk(1);
    scheduler_->active_[2] = mk(2);
    scheduler_->active_[3] = mk(3);
    scheduler_->active_order_ = {1, 2, 3};

    scheduler_->build_scheduled_batch();

    ASSERT_EQ(scheduler_->active_order_.size(), 3u);
    EXPECT_EQ(scheduler_->active_order_[0], 1u);
    EXPECT_EQ(scheduler_->active_order_[1], 2u);
    EXPECT_EQ(scheduler_->active_order_[2], 3u);
}

}  // namespace
}  // namespace server
}  // namespace ccinfer
