#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>

#include "backend/backend.h"
#include "base/channel.h"
#include "base/error.h"
#include "base/types.h"
#include "block/block_storage.h"
#include "config/engine_config.h"
#include "config/model_config.h"
#include "facade/log.h"
#include "state/state_pool.h"
#include "worker/model_runner.h"

namespace ccinfer {

namespace asio = boost::asio;

class Worker {
public:
    class BatchTranslator {
    public:
        BatchTranslator(Backend& backend, int block_size, StatePool* state_pool = nullptr);
        Result<PhysicalBatch> translate(const ScheduledBatch& batch) const;

    private:
        Backend& backend_;
        int block_size_;
        StatePool* state_pool_;
    };

    explicit Worker(asio::io_context& io);
    ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    Result<void> init(const std::string& model_path, const ModelConfig& model,
                      const EngineConfig& engine);
    void shutdown();

    Result<BatchFuture> enqueue_execute_batch(ScheduledBatch batch);

    // Releases the sequence's active GDN state. Callers (Scheduler cleanup)
    // must only call this after the sequence is terminal/failed/preempted and
    // its authoritative execution_leases has reached zero.
    Result<void> release_sequence(SequenceId seq);

private:
    struct PendingBatch {
        ScheduledBatch batch;
        std::shared_ptr<BatchChannel> chan;
    };

    struct ResolvedBatch {
        ScheduledBatch batch;
        std::vector<std::size_t> original_indices;
        std::vector<WorkItemResult> stale_results;
    };

    void worker_loop();
    void process_command(PendingBatch pending);
    void process_batch(PendingBatch pending);

    Result<void> init_resources(const std::string& model_path, const ModelConfig& model,
                                const EngineConfig& engine);
    void reset_resources();

    ResolvedBatch resolve_batch(const ScheduledBatch& batch) const;
    void resolve(const std::shared_ptr<BatchChannel>& channel, Result<BatchResult> result);

    asio::io_context& io_;

    std::deque<PendingBatch> queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    // Execution-only handoff state. Scheduler policy, block ownership and
    // request lifecycle remain independent of these maps.
    std::unordered_map<SequenceId, int32_t> latest_tokens_;
    std::unordered_set<SequenceId> failed_sequences_;
    // Active GDN state release is driven by the Scheduler's authoritative
    // SequenceReservation::execution_leases via Executor::release_sequence.
    // Worker itself does not keep a second lease authority.
    bool initialized_ = false;
    EngineConfig engine_config_;

    std::unique_ptr<Backend> backend_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<BlockStorage> block_storage_;
    std::unique_ptr<StatePool> state_pool_;
};

}  // namespace ccinfer
