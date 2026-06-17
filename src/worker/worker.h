#pragma once

#include <atomic>
#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/channel.h"
#include "base/result.h"
#include "base/types.h"
#include "backend/default_backend.h"
#include "base/execution.h"

namespace ccinfer {

namespace asio = boost::asio;

class KVCacheManager;
class Model;

class Worker {
public:
    explicit Worker(asio::io_context& io);
    ~Worker();

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    Result<void> init(const std::string& model_path);
    void shutdown();

    Result<CreateSequenceResult> prepare_sequence_resources(
        SequenceId seq_id, const std::vector<int32_t>& prompt_tokens, int max_context_len);
    Result<SuspendSequenceResult> reset_sequence_resources(
        SequenceId seq_id, const std::vector<int32_t>& prompt_tokens, int max_context_len);
    Result<void> release_sequence_resources(SequenceId seq_id);
    void enqueue_execute_batch(ScheduledBatch batch, std::vector<SequenceSnapshot> sequences,
                               std::shared_ptr<BatchChannel> chan);

    DeviceCapacity capacity() const;

private:
    struct PendingBatch {
        ScheduledBatch batch;
        std::vector<SequenceSnapshot> sequences;
        std::shared_ptr<BatchChannel> chan;
    };

    // --- Worker thread ---
    void worker_loop();
    void process_batch(PendingBatch pending);

    // --- Resource lifecycle ---
    Result<void> init_resources(const std::string& model_path);
    void reset_resources();

    // --- Sequence helpers ---
    using BlockTableIter = std::unordered_map<SequenceId, BlockTable>::iterator;
    Result<void> release_sequence_blocks(BlockTableIter it);

    // --- Capacity sync ---
    void sync_capacity();

    // --- Dummy output (no model loaded yet) ---
    BatchResult generate_dummy_result(const ScheduledBatch& batch);

    // --- Completion via io_context ---
    template <typename ChanPtr, typename T>
    void resolve(ChanPtr& chan_ptr, Result<T> result);

    // --- State ---
    asio::io_context& io_;

    std::deque<PendingBatch> queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    std::mutex resource_mutex_;
    std::unordered_map<SequenceId, BlockTable> block_tables_;

    // --- Cached capacity (synced from KVCacheManager::stats()) ---
    std::atomic<int> active_sequences_{0};
    std::atomic<int> free_blocks_{0};
    std::atomic<int> max_blocks_{0};
    std::atomic<int> block_size_{0};
    std::atomic<int> block_active_{0};
    std::atomic<int> block_cached_idle_{0};
    std::atomic<uint64_t> prefix_lookup_hits_{0};
    std::atomic<uint64_t> prefix_lookup_misses_{0};
    std::atomic<uint64_t> prefix_evictions_{0};
    std::atomic<uint64_t> prefix_cached_blocks_{0};
    static constexpr int kMaxSequences = 64;

    bool initialized_ = false;

    // --- Device resources protected by resource_mutex_ ---
    std::unique_ptr<DefaultBackend> backend_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<KVCacheManager> kv_mgr_;
};

// --- Template definitions ---

template <typename ChanPtr, typename T>
void Worker::resolve(ChanPtr& chan_ptr, Result<T> result) {
    asio::post(io_, [chan_ptr, result = std::move(result)]() mutable {
        chan_ptr->try_send(boost::system::error_code{}, std::move(result));
    });
}

}  // namespace ccinfer
