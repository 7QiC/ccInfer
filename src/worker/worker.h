#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/asio/io_context.hpp>
#include <boost/asio/post.hpp>

#include "backend/backend.h"
#include "base/channel.h"
#include "base/result.h"
#include "base/types.h"
#include "config/config.h"
#include "executor/execution.h"
#include "facade/log.h"

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

    Result<void> init(const Config& config);
    void shutdown();

    Result<std::shared_ptr<AdmitSequenceChannel>> enqueue_admit_sequence(
        SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len,
        SequenceInitialState initial_state = {});
    Result<std::shared_ptr<SuspendSequenceChannel>> enqueue_suspend_sequence(
        SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len);
    Result<std::shared_ptr<VoidChannel>> enqueue_release_sequence(SequenceId seq_id);
    Result<std::shared_ptr<VoidChannel>> enqueue_abort_sequence(SequenceId seq_id);
    Result<BatchFuture> enqueue_execute_batch(ScheduledBatch batch);

    Capacity capacity() const;

private:
    struct AdmitCommand {
        SequenceId seq_id;
        std::vector<int32_t> prompt_tokens;
        int max_context_len;
        SequenceInitialState initial_state;
        std::shared_ptr<AdmitSequenceChannel> channel;
    };

    struct SuspendCommand {
        SequenceId seq_id;
        std::vector<int32_t> prompt_tokens;
        int max_context_len;
        std::shared_ptr<SuspendSequenceChannel> channel;
    };

    struct ReleaseCommand {
        SequenceId seq_id;
        std::shared_ptr<VoidChannel> channel;
    };

    struct AbortCommand {
        SequenceId seq_id;
        std::shared_ptr<VoidChannel> channel;
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

    using PendingCommand =
        std::variant<AdmitCommand, SuspendCommand, ReleaseCommand, AbortCommand, PendingBatch>;
    using SequenceRegistry = std::unordered_map<SequenceId, SequenceState>;

    void worker_loop();
    void process_command(PendingCommand command);
    void process_admit(AdmitCommand command);
    void process_suspend(SuspendCommand command);
    void process_release(ReleaseCommand command);
    void process_abort(AbortCommand command);
    void process_batch(PendingBatch pending);

    Result<void> init_resources(const Config& config);
    void reset_resources();

    Result<AdmitSequenceResult> admit_sequence_resources(SequenceId seq_id,
                                                         const std::vector<int32_t>& prompt_tokens,
                                                         int max_context_len,
                                                         SequenceInitialState initial_state);
    Result<SuspendSequenceResult> reset_sequence_resources(
        SequenceId seq_id, const std::vector<int32_t>& prompt_tokens, int max_context_len);
    Result<void> release_sequence_resources(SequenceId seq_id);

    void sync_capacity();

    SequenceRegistry build_sequence_states(const ScheduledBatch& batch) const;
    Result<ResolvedBatch> resolve_batch(const ScheduledBatch& batch) const;
    Result<std::vector<SequenceDelta>> build_deltas(const SequenceRegistry& before,
                                                    const SequenceRegistry& after);
    static Result<void> apply_sequence_deltas(SequenceRegistry& states,
                                              const std::vector<SequenceDelta>& deltas);
    static Result<void> apply_sampled_progress(SequenceRegistry& states, const BatchResult& batch);

    WorkerBatchResult generate_dummy_batch_result(const ScheduledBatch& batch) const;

    template <typename ChanPtr, typename T>
    void resolve(ChanPtr& chan_ptr, Result<T> result);

    asio::io_context& io_;

    std::deque<PendingCommand> queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::size_t in_flight_batches_ = 0;

    std::mutex resource_mutex_;
    std::unordered_map<SequenceId, SequenceState> request_states_;

    // Cached capacity, synced from KVCacheManager::stats().
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
    bool initialized_ = false;
    EngineConfig engine_config_;

    // Device resources protected by resource_mutex_.
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<KVCacheManager> kv_mgr_;
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
