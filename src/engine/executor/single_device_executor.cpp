#include "engine/executor/single_device_executor.h"

#include <utility>

#include "engine/worker/worker.h"

namespace ccinfer {
namespace engine {

SingleDeviceExecutor::SingleDeviceExecutor(boost::asio::io_context& io)
    : worker_(std::make_unique<Worker>(io)) {}

SingleDeviceExecutor::~SingleDeviceExecutor() { shutdown(); }

Result<void> SingleDeviceExecutor::init(const std::string& model_path) {
    return worker_->init(model_path);
}

void SingleDeviceExecutor::shutdown() { worker_->shutdown(); }

void SingleDeviceExecutor::enqueue_create_sequence(std::vector<int32_t> prompt_tokens,
                                                   int max_context_len,
                                                   std::shared_ptr<SeqIdChannel> chan) {
    worker_->enqueue_create_sequence(std::move(prompt_tokens), max_context_len, std::move(chan));
}

void SingleDeviceExecutor::enqueue_release_sequence(SequenceId seq_id,
                                                    std::shared_ptr<VoidChannel> chan) {
    worker_->enqueue_release_sequence(seq_id, std::move(chan));
}

void SingleDeviceExecutor::enqueue_abort_sequence(SequenceId seq_id,
                                                  std::shared_ptr<VoidChannel> chan) {
    worker_->enqueue_abort_sequence(seq_id, std::move(chan));
}

void SingleDeviceExecutor::enqueue_execute_batch(ScheduledBatch batch,
                                                 std::shared_ptr<BatchChannel> chan) {
    worker_->enqueue_execute_batch(std::move(batch), std::move(chan));
}

EngineCapacity SingleDeviceExecutor::capacity() const {
    auto cap = worker_->capacity();
    return EngineCapacity{cap.max_sequences, cap.active_sequences, cap.free_blocks, cap.max_blocks,
                          cap.block_size, cap.block_active, cap.block_cached_idle,
                          cap.prefix_lookup_hits, cap.prefix_lookup_misses,
                          cap.prefix_evictions, cap.prefix_cached_blocks};
}

std::unique_ptr<Executor> Executor::create(boost::asio::io_context& io) {
    return std::make_unique<SingleDeviceExecutor>(io);
}

}  // namespace engine
}  // namespace ccinfer
