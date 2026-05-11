#pragma once

#include <memory>
#include <string>
#include <vector>

#include "engine/executor/executor.h"

namespace ccinfer {
namespace engine {

class Worker;

class SingleDeviceExecutor final : public Executor {
public:
    explicit SingleDeviceExecutor(boost::asio::io_context& io);
    ~SingleDeviceExecutor() override;

    Result<void> init(const std::string& model_path) override;
    void shutdown() override;

    void enqueue_create_sequence(std::vector<int32_t> prompt_tokens, int max_context_len,
                                 std::shared_ptr<SeqIdChannel> chan) override;
    void enqueue_release_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan) override;
    void enqueue_abort_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan) override;
    void enqueue_execute_batch(ScheduledBatch batch, std::shared_ptr<BatchChannel> chan) override;

    EngineCapacity capacity() const override;

private:
    std::unique_ptr<Worker> worker_;
};

}  // namespace engine
}  // namespace ccinfer
