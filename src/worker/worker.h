#pragma once

#include <atomic>
#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
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

    void enqueue_create_sequence(std::vector<int32_t> prompt_tokens, int max_context_len,
                                 std::shared_ptr<SeqIdChannel> chan);
    void enqueue_suspend_sequence(SequenceId seq_id, std::vector<int32_t> prompt_tokens,
                                  int max_context_len, std::shared_ptr<SuspendChannel> chan);
    void enqueue_release_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan);
    void enqueue_abort_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan);
    void enqueue_execute_batch(ScheduledBatch batch, std::shared_ptr<BatchChannel> chan);

    DeviceCapacity capacity() const;

private:
    // --- Queue entries ---
    struct CmdInitResources {
        std::string model_path;
        std::shared_ptr<std::promise<Result<void>>> promise;
    };
    struct CmdCreateSequence {
        std::vector<int32_t> prompt_tokens;
        int max_context_len;
        std::shared_ptr<SeqIdChannel> chan;
    };
    struct CmdSuspendSequence {
        SequenceId seq_id;
        std::vector<int32_t> prompt_tokens;
        int max_context_len;
        std::shared_ptr<SuspendChannel> chan;
    };
    struct CmdReleaseSequence {
        SequenceId seq_id;
        std::shared_ptr<VoidChannel> chan;
    };
    struct CmdAbortSequence {
        SequenceId seq_id;
        std::shared_ptr<VoidChannel> chan;
    };
    using ControlCmd = std::variant<CmdInitResources, CmdCreateSequence, CmdSuspendSequence,
                                    CmdReleaseSequence, CmdAbortSequence>;

    struct PendingBatch {
        ScheduledBatch batch;
        std::shared_ptr<BatchChannel> chan;
    };

    using QueueEntry = std::variant<ControlCmd, PendingBatch>;

    // --- Worker thread ---
    void worker_loop();
    void process_control(ControlCmd& cmd);
    void process_batch(PendingBatch pending);

    // --- Resource lifecycle (runs on worker thread) ---
    void init_resources(const std::string& model_path,
                        std::shared_ptr<std::promise<Result<void>>> promise);

    // --- Sequence helpers ---
    using SeqMapIter = std::unordered_map<SequenceId, SequenceState>::iterator;
    Result<void> release_sequence_state(SeqMapIter it);

    // --- Capacity sync ---
    void sync_capacity();

    // --- Dummy output (no model loaded yet) ---
    BatchResult generate_dummy_result(const ScheduledBatch& batch);

    // --- Completion via io_context ---
    template <typename ChanPtr, typename T>
    void resolve(ChanPtr& chan_ptr, Result<T> result);

    // --- State ---
    asio::io_context& io_;

    std::deque<QueueEntry> queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    std::unordered_map<SequenceId, SequenceState> sequences_;
    std::atomic<SequenceId> next_seq_id_{1};

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

    // --- Resources (owned by worker thread after init_resources) ---
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
