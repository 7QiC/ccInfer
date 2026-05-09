#pragma once

#include <cuda_bf16.h>

#include <atomic>
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

#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>

#include "common/channel.h"
#include "common/result.h"
#include "common/types.h"
#include "engine/common/backend_def.h"
#include "engine/common/types.h"

namespace ccinfer {
namespace engine {

namespace asio = boost::asio;

struct DeviceCapacity {
    int max_sequences = 0;
    int active_sequences = 0;
    int free_blocks = 0;
    int max_blocks = 0;
};

class KVCacheManager;
class Model;
class KVCacheStorage;

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
    struct CmdReleaseSequence {
        SequenceId seq_id;
        std::shared_ptr<VoidChannel> chan;
    };
    struct CmdAbortSequence {
        SequenceId seq_id;
        std::shared_ptr<VoidChannel> chan;
    };
    using ControlCmd =
        std::variant<CmdInitResources, CmdCreateSequence, CmdReleaseSequence, CmdAbortSequence>;

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
    void release_sequence_state(SeqMapIter it);

    // --- Dummy output (no model loaded yet) ---
    BatchResult generate_dummy_result(const ScheduledBatch& batch);

    // --- Completion via io_context ---
    template <typename ChanPtr, typename... Args>
    void resolve(ChanPtr& chan_ptr, ErrorCode ec, Args&&... args);

    // --- State ---
    asio::io_context& io_;

    std::deque<QueueEntry> queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    std::unordered_map<SequenceId, SequenceState> sequences_;
    std::atomic<SequenceId> next_seq_id_{1};

    // --- Cached capacity ---
    std::atomic<int> active_sequences_{0};
    std::atomic<int> free_blocks_{0};
    int max_blocks_ = 0;
    static constexpr int kMaxSequences = 64;

    // --- Resources (owned by worker thread after init_resources) ---
    std::unique_ptr<DefaultBackend> backend_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<KVCacheManager> kv_mgr_;
    std::unique_ptr<KVCacheStorage> kv_storage_;
};

// --- Template definitions ---

template <typename ChanPtr, typename... Args>
void Worker::resolve(ChanPtr& chan_ptr, ErrorCode ec, Args&&... args) {
    asio::post(io_, [chan_ptr, ec, ... args = std::move(args)]() mutable {
        chan_ptr->try_send(ec, std::move(args)...);
    });
}

}  // namespace engine
}  // namespace ccinfer
