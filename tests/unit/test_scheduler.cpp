#include <algorithm>
#include <chrono>
#include <deque>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/use_future.hpp>
#include <gtest/gtest.h>

#include "scheduler/scheduler.h"

#include "common/channel.h"

namespace ccinfer {

struct SchedulerTestAccess {
    static auto schedule_step(Scheduler& s) { return s.schedule_step(); }
    static auto update_from_output(Scheduler& s, const ScheduledBatch& b, const BatchResult& r) {
        return s.update_from_output(b, r);
    }
    static void submit_on_scheduler_thread(Scheduler& s, SchedulerRequest req) {
        s.submit_on_scheduler_thread(std::move(req));
    }
    static auto& accepting(Scheduler& s) { return s.accepting_; }
    static auto& engine_config(Scheduler& s) { return s.engine_config_; }
    static auto& requests(Scheduler& s) { return s.requests_; }
    static auto& waiting(Scheduler& s) { return s.waiting_; }
    static auto& running(Scheduler& s) { return s.running_; }
    static auto& by_seq_id(Scheduler& s) { return s.by_seq_id_; }
};

namespace {

class FakeExecutor final : public Executor {
public:
    explicit FakeExecutor(boost::asio::io_context& io) : io_(io) {}

    Result<void> init(const Config&) override { return {}; }
    void shutdown() override {}

    asio::awaitable<Result<AdmitSequenceResult>> admit_sequence(
        std::vector<int32_t> prompt, int, SequenceInitialState = {}) override {
        co_return AdmitSequenceResult{next_id_++, static_cast<int>(prompt.size())};
    }

    asio::awaitable<Result<SuspendSequenceResult>> suspend_sequence(SequenceId,
                                                                    std::vector<int32_t>,
                                                                    int) override {
        co_return SuspendSequenceResult{0};
    }

    asio::awaitable<Result<void>> release_sequence(SequenceId) override {
        co_return Result<void>{};
    }
    asio::awaitable<Result<void>> abort_sequence(SequenceId) override { co_return Result<void>{}; }

    Result<BatchFuture> execute_batch(ScheduledBatch batch) override {
        ++execute_calls_;
        auto future = std::make_shared<BatchChannel>(io_.get_executor(), 1);
        batches_.push_back(std::move(batch));
        futures_.push_back(future);
        max_queued_batches_ = std::max(max_queued_batches_, batches_.size());
        if (auto_complete_) asio::post(io_, [this] { complete_oldest(); });
        return future;
    }

    asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) override {
        auto [ec, result] = co_await future->async_receive(asio::as_tuple(asio::deferred));
        if (ec) co_return std::unexpected(ErrorCode::ChannelClosed);
        if (!result) co_return std::unexpected(result.error());
        co_return std::move(result->batch);
    }

    Capacity capacity() const override { return cap_; }

    void complete_oldest() {
        ASSERT_FALSE(batches_.empty());
        auto batch = std::move(batches_.front());
        batches_.pop_front();
        auto future = std::move(futures_.front());
        futures_.pop_front();

        WorkerBatchResult result;
        result.batch.batch_id = batch.batch_id;
        for (std::size_t i = 0; i < batch.items.size(); ++i) {
            WorkItemResult item_result;
            item_result.item_index = static_cast<int>(i);
            item_result.sampled_tokens = {42};
            std::visit(
                [&](const auto& work) {
                    item_result.seq_id = work.seq_id;
                    using T = std::decay_t<decltype(work)>;
                    if constexpr (std::is_same_v<T, PrefillChunk>) {
                        item_result.kind = WorkKind::PrefillChunk;
                        item_result.tokens_consumed = work.prompt_span.length;
                    } else {
                        item_result.kind = WorkKind::DecodeOneToken;
                        item_result.tokens_consumed = 1;
                    }
                },
                batch.items[i]);
            result.batch.items.push_back(std::move(item_result));
        }
        future->try_send(boost::system::error_code{}, std::move(result));
    }

    int execute_calls() const { return execute_calls_; }
    std::size_t queued_batches() const { return batches_.size(); }
    std::size_t max_queued_batches() const { return max_queued_batches_; }

    boost::asio::io_context& io_;
    Capacity cap_{64, 0, 1024, 1024, 16, 0, 0, 0, 0, 0, 0};
    SequenceId next_id_{1};
    int execute_calls_{0};
    bool auto_complete_{false};
    std::size_t max_queued_batches_{0};
    std::deque<ScheduledBatch> batches_;
    std::deque<BatchFuture> futures_;
};

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.max_token_budget = 4096;
        config_.max_seq_prefill_tokens = 512;
        scheduler_ = std::make_unique<Scheduler>(io_, executor_, config_);
    }

    boost::asio::io_context io_;
    FakeExecutor executor_{io_};
    EngineConfig config_;
    std::unique_ptr<Scheduler> scheduler_;

    ScheduledBatch schedule_step() {
        io_.restart();
        auto future = asio::co_spawn(io_, SchedulerTestAccess::schedule_step(*scheduler_), asio::use_future);
        io_.run();
        return future.get();
    }
};

RequestState make_decode(SequenceId id) {
    RequestState state;
    state.scheduling.emplace();
    state.scheduling->seq_id = id;
    state.scheduling->cursor.phase = GenerationPhase::Decode;
    state.prompt_tokens = {1};
    state.max_context_len = 64;
    state.sampling.max_tokens = 10;
    return state;
}

RequestState& add_running(Scheduler& scheduler, RequestState state) {
    if (state.request_id.empty()) {
        state.request_id = "seq-" + std::to_string(state.scheduling->seq_id);
    }
    auto request = std::make_shared<RequestState>(std::move(state));
    SchedulerTestAccess::requests(scheduler)[request->request_id] = request;
    SchedulerTestAccess::running(scheduler).push_back(request);
    SchedulerTestAccess::by_seq_id(scheduler)[request->scheduling->seq_id] = request;
    return *request;
}

TEST_F(SchedulerTest, RunningSetAllowsSameSequenceAcrossInFlightBatches) {
    auto& first = add_running(*scheduler_, make_decode(1));
    auto& second = add_running(*scheduler_, make_decode(2));

    auto batch = schedule_step();
    ASSERT_EQ(batch.items.size(), 2u);
    EXPECT_EQ(first.scheduling->reservation.execution_leases, 1);
    EXPECT_EQ(second.scheduling->reservation.execution_leases, 1);
    EXPECT_EQ(SchedulerTestAccess::running(*scheduler_).size(), 2u);

    auto next = schedule_step();
    ASSERT_EQ(next.items.size(), 2u);
    EXPECT_EQ(first.scheduling->reservation.execution_leases, 2);
    EXPECT_EQ(second.scheduling->reservation.execution_leases, 2);
}

TEST_F(SchedulerTest, AdmissionKeepsOneCanonicalRequestState) {
    SchedulerTestAccess::accepting(*scheduler_).store(true);
    SchedulerRequest request;
    request.request_id = "canonical";
    request.prompt_tokens = {1, 2, 3};
    request.sampling.max_tokens = 4;
    auto channel = std::make_shared<TokenChannel>(io_.get_executor(), 16);
    request.sink.executor = io_.get_executor();
    request.sink.channel = channel;
    SchedulerTestAccess::submit_on_scheduler_thread(*scheduler_, std::move(request));

    ASSERT_EQ(SchedulerTestAccess::requests(*scheduler_).size(), 1u);
    const auto request_ptr = SchedulerTestAccess::waiting(*scheduler_).front();
    auto batch = schedule_step();

    ASSERT_EQ(batch.items.size(), 1u);
    EXPECT_TRUE(SchedulerTestAccess::waiting(*scheduler_).empty());
    ASSERT_EQ(SchedulerTestAccess::running(*scheduler_).size(), 1u);
    EXPECT_EQ(SchedulerTestAccess::running(*scheduler_).front(), request_ptr);
    EXPECT_EQ(SchedulerTestAccess::requests(*scheduler_).at("canonical"), request_ptr);
    EXPECT_TRUE(request_ptr->scheduling.has_value());
}

TEST_F(SchedulerTest, PerSequencePrefillCapBoundsOneBatch) {
    SchedulerTestAccess::engine_config(*scheduler_).max_token_budget = 2048;
    SchedulerTestAccess::engine_config(*scheduler_).max_seq_prefill_tokens = 256;
    RequestState state;
    state.scheduling.emplace();
    state.scheduling->seq_id = 1;
    state.prompt_tokens = std::vector<int32_t>(4096, 1);
    state.max_context_len = 4096;
    state.sampling.max_tokens = 10;
    add_running(*scheduler_, std::move(state));

    auto batch = schedule_step();
    ASSERT_EQ(batch.items.size(), 1u);
    EXPECT_EQ(std::get<PrefillChunk>(batch.items.front()).prompt_span.length, 256);
}

TEST_F(SchedulerTest, DecodeAndPrefillShareOneTokenBudget) {
    SchedulerTestAccess::engine_config(*scheduler_).max_token_budget = 3;
    SchedulerTestAccess::engine_config(*scheduler_).max_seq_prefill_tokens = 2;
    auto decode = make_decode(1);
    RequestState prefill;
    prefill.scheduling.emplace();
    prefill.scheduling->seq_id = 2;
    prefill.prompt_tokens = {1, 2, 3, 4};
    prefill.max_context_len = 64;
    prefill.sampling.max_tokens = 10;
    add_running(*scheduler_, std::move(decode));
    add_running(*scheduler_, std::move(prefill));

    auto batch = schedule_step();
    ASSERT_EQ(batch.items.size(), 2u);
    EXPECT_EQ(std::get<DecodeOneToken>(batch.items[0]).seq_id, 1u);
    EXPECT_EQ(std::get<PrefillChunk>(batch.items[1]).prompt_span.length, 2);
}

TEST_F(SchedulerTest, NonBlockingExecutorKeepsMultipleBatchFutures) {
    ScheduledBatch first;
    first.batch_id = 1;
    first.items.push_back(DecodeOneToken{1, 7, 0, true});
    ScheduledBatch second;
    second.batch_id = 2;
    second.items.push_back(DecodeOneToken{2, 8, 0, true});

    auto first_future = executor_.execute_batch(first);
    auto second_future = executor_.execute_batch(second);
    ASSERT_TRUE(first_future.has_value());
    ASSERT_TRUE(second_future.has_value());
    EXPECT_EQ(executor_.execute_calls(), 2);
    EXPECT_EQ(executor_.queued_batches(), 2u);
}

TEST_F(SchedulerTest, EngineCoreDispatchesUpToConcurrentBatchLimit) {
    SchedulerTestAccess::engine_config(*scheduler_).max_concurrent_batches = 2;
    SchedulerTestAccess::engine_config(*scheduler_).max_running_requests = 4;
    SchedulerTestAccess::engine_config(*scheduler_).max_token_budget = 1;
    SchedulerTestAccess::engine_config(*scheduler_).max_seq_prefill_tokens = 1;
    executor_.auto_complete_ = true;

    scheduler_->start();
    for (int i = 0; i < 4; ++i) {
        SchedulerRequest request;
        request.request_id = "req-" + std::to_string(i);
        request.prompt_tokens = {1};
        request.sampling.max_tokens = 1;
        scheduler_->submit(std::move(request));
    }

    io_.run_for(std::chrono::milliseconds(20));
    EXPECT_GE(executor_.execute_calls(), 2);
    EXPECT_EQ(executor_.max_queued_batches(), 2u);

    auto shutdown = scheduler_->shutdown_async();
    io_.run_for(std::chrono::milliseconds(20));
    EXPECT_EQ(shutdown.wait_for(std::chrono::seconds(0)), std::future_status::ready);
}

TEST_F(SchedulerTest, BootstrapTransitionsToDecodeAfterFirstSample) {
    SchedulerTestAccess::accepting(*scheduler_).store(true);
    SchedulerRequest request;
    request.request_id = "bootstrap";
    request.prompt_tokens = {1, 2, 3};
    request.sampling.max_tokens = 4;
    auto channel = std::make_shared<TokenChannel>(io_.get_executor(), 16);
    request.sink.executor = io_.get_executor();
    request.sink.channel = channel;
    SchedulerTestAccess::submit_on_scheduler_thread(*scheduler_, std::move(request));

    auto batch = schedule_step();
    ASSERT_EQ(batch.items.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<DecodeOneToken>(batch.items.front()));
    const auto& first = std::get<DecodeOneToken>(batch.items.front());
    EXPECT_FALSE(first.write_kv);
    EXPECT_FALSE(first.late_bind);
    EXPECT_EQ(first.input_token, 3);

    BatchResult result;
    result.batch_id = batch.batch_id;
    WorkItemResult wr;
    wr.item_index = 0;
    wr.seq_id = first.seq_id;
    wr.kind = WorkKind::DecodeOneToken;
    wr.sampled_tokens = {42};
    wr.tokens_consumed = 1;
    result.items.push_back(std::move(wr));

    io_.restart();
    auto update_future =
        asio::co_spawn(io_, SchedulerTestAccess::update_from_output(*scheduler_, batch, result),
                       asio::use_future);
    io_.run();
    update_future.get();

    ASSERT_TRUE(SchedulerTestAccess::running(*scheduler_).front()->scheduling.has_value());
    auto& st = *SchedulerTestAccess::running(*scheduler_).front();
    EXPECT_EQ(st.scheduling->cursor.phase, GenerationPhase::Decode);
    EXPECT_EQ(st.generated_tokens.size(), 1u);

    auto next = schedule_step();
    ASSERT_EQ(next.items.size(), 1u);
    ASSERT_TRUE(std::holds_alternative<DecodeOneToken>(next.items.front()));
    const auto& second = std::get<DecodeOneToken>(next.items.front());
    EXPECT_TRUE(second.write_kv);
    EXPECT_TRUE(second.late_bind);
}

}  // namespace
}  // namespace ccinfer
