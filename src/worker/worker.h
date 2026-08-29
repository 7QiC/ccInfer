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
#include "cache/block_storage.h"
#include "common/channel.h"
#include "common/error_code.h"
#include "common/physical_batch.h"
#include "common/types.h"
#include "config/config.h"
#include "facade/log.h"

namespace ccinfer {

namespace asio = boost::asio;

class Model;
struct WorkerTestAccess;

class Worker {
public:
    explicit Worker(asio::io_context& io);
    ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    Result<void> init(const Config& config);
    void shutdown();

    Result<BatchFuture> enqueue_execute_batch(ScheduledBatch batch);

private:
    friend struct WorkerTestAccess;

    class BatchTranslator {
    public:
        BatchTranslator(Backend& backend, int block_size);
        Result<PhysicalBatch> translate(const ScheduledBatch& batch) const;

    private:
        Backend& backend_;
        int block_size_;
    };

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

    Result<void> init_resources(const Config& config);
    void reset_resources();

    ResolvedBatch resolve_batch(const ScheduledBatch& batch) const;

    template <typename ChanPtr, typename T>
    void resolve(ChanPtr& chan_ptr, Result<T> result);

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
    bool initialized_ = false;
    EngineConfig engine_config_;

    std::unique_ptr<Backend> backend_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<BlockStorage> block_storage_;
};

template <typename ChanPtr, typename T>
void Worker::resolve(ChanPtr& chan_ptr, Result<T> result) {
    asio::post(io_, [chan_ptr, result = std::move(result)]() mutable {
        chan_ptr->async_send(
            boost::system::error_code{}, std::move(result), [](boost::system::error_code ec) {
                if (ec) {
                    ccLog::warn("worker completion channel failed ec={}", ec.value());
                }
            });
    });
}

}  // namespace ccinfer
