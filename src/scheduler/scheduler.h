#pragma once

#include <atomic>
#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/steady_timer.hpp>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/request.h"
#include "executor/executor.h"

namespace ccinfer {

namespace asio = boost::asio;

// Scheduler-owned request state — only accessed on scheduler_io thread.
struct SchedulerRequestState {
    std::string request_id;
    SequenceId seq_id = 0;

    std::vector<int32_t> initial_prompt_tokens;
    std::vector<int32_t> prompt_tokens;
    int prefill_cursor = 0;  // number of prompt tokens already consumed / committed
    bool prefill_done = false;
    bool suspended = false;
    int generated_in_prompt = 0;

    int32_t last_token = -1;   // next decode input; sampled but not yet in KV cache
    int tokens_generated = 0;  // output tokens already sent to client
    std::vector<int32_t> generated_tokens;
    bool finished = false;
    bool cancelled = false;

    SamplingParams sampling;
    int max_context_len = 2048;

    TokenSink sink;
};

// Scheduler must outlive the detached do_work() coroutine.
// Server shutdown must call shutdown() and keep io_context running
// until do_work() exits.
//
// Public API (submit, cancel, start, shutdown, shutdown_async) is thread-safe.
// All internal state is accessed only on the scheduler_io thread.

class Scheduler {
public:
    Scheduler(asio::io_context& io, Executor& executor);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Thread-safe — callable from any thread.
    void submit(SchedulerRequest req);
    void cancel(std::string request_id);
    void start();
    EngineCapacity capacity() const;

    // Fire-and-forget shutdown.  Prefer shutdown_async() for waitable shutdown.
    void shutdown();

    // Initiates shutdown; returns a future that is set when do_work() has
    // completed its cleanup (fail-all-pending + cleanup-all-active).
    std::shared_future<void> shutdown_async();

private:
    using StatePtr = std::shared_ptr<SchedulerRequestState>;
    using SeqMap = std::unordered_map<SequenceId, StatePtr>;

    // Must be called on scheduler_io thread.
    void submit_on_scheduler_thread(SchedulerRequest req);
    void cancel_on_scheduler_thread(const std::string& request_id);
    void wake_on_scheduler_thread();
    void complete_shutdown_on_scheduler_thread();

    asio::awaitable<void> do_work();
    asio::awaitable<void> drain_pending();
    ScheduledBatch build_scheduled_batch();
    asio::awaitable<void> apply_and_push(const ScheduledBatch& batch, const BatchResult& result);
    asio::awaitable<void> cleanup_cancelled_or_finished();
    asio::awaitable<void> fail_batch(const ScheduledBatch& batch, ErrorCode err);
    asio::awaitable<void> suspend_sequence_for_replay(StatePtr& state);
    void fail_all_pending_on_scheduler_thread(ErrorCode err);
    asio::awaitable<void> cleanup_all_active(ErrorCode shutdown_err);
    asio::awaitable<void> wait_for_work();

    void filter_active_order();
    void reorder_after_batch(const std::vector<SequenceId>& included);

    bool send_event(const TokenSink& sink, Result<GeneratedToken> result);
    bool send_token_event(StatePtr& state);
    bool send_terminal_event(StatePtr& state);
    bool send_error_event(StatePtr& state, ErrorCode err);

    asio::io_context& io_;
    Executor& executor_;
    asio::steady_timer idle_timer_;

    // Only accessed on scheduler_io thread — no mutex needed.
    std::vector<SchedulerRequest> pending_requests_;
    std::unordered_set<std::string> pending_or_creating_ids_;
    std::unordered_set<std::string> cancelled_pending_ids_;
    SeqMap active_;
    std::deque<SequenceId> active_order_;
    std::unordered_map<std::string, StatePtr> by_request_id_;
    std::unordered_map<SequenceId, std::string> seq_to_request_id_;

    std::vector<SequenceId> budget_blocked_{};  // items skipped due to KV budget
    std::atomic<bool> running_{false};
    uint64_t next_batch_id_{1};

    // Shutdown synchronisation.
    std::mutex shutdown_mutex_;
    std::unique_ptr<std::promise<void>> shutdown_promise_;
    std::shared_future<void> shutdown_future_;
    bool shutdown_done_sent_{false};
};

}  // namespace ccinfer
