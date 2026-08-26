#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "executor/executor.h"

namespace ccinfer {

class Worker;

class SingleDeviceExecutor final : public Executor {
public:
    explicit SingleDeviceExecutor(boost::asio::io_context& io);
    ~SingleDeviceExecutor() override;

    Result<void> init(const Config& config) override;
    void shutdown() override;

    asio::awaitable<Result<AdmitSequenceResult>> admit_sequence(
        std::vector<int32_t> prompt_tokens, int max_context_len,
        SequenceInitialState initial_state = {}) override;
    asio::awaitable<Result<void>> release_sequence(SequenceId seq_id) override;
    asio::awaitable<Result<void>> abort_sequence(SequenceId seq_id) override;
    Result<BatchFuture> execute_batch(ScheduledBatch batch) override;
    asio::awaitable<Result<BatchResult>> collect_batch(BatchFuture future) override;

    Capacity capacity() const override;

private:
    std::unordered_map<SequenceId, SequenceSnapshot> sequences_;
    SequenceId next_seq_id_{1};
    std::unique_ptr<Worker> worker_;
};

}  // namespace ccinfer
