#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <future>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/steady_timer.hpp>

#include "cache/block_pool.h"
#include "common/types.h"
#include "config/config.h"
#include "executor/executor.h"

namespace ccinfer {

namespace asio = boost::asio;

enum class GenerationPhase : uint8_t { Prefill, ReplayPrefill, Bootstrap, Decode };

struct SequenceReservation {
    // Number of scheduled work items that still own an execution lifetime
    // lease. This is not a scheduling mutex.
    int execution_leases = 0;
    int reserved_prompt_tokens = 0;
    int reserved_generation_tokens = 0;
    int reserved_kv_tokens = 0;
};

// Scheduler-side phase cursor. Physical execution is coordinated by Executor;
// Worker does not own a logical sequence state.
struct SequenceScheduleCursor {
    GenerationPhase phase = GenerationPhase::Prefill;
    int prefill_cursor = 0;
    int generated_tokens_in_prompt = 0;
};

struct SequenceSchedulingState {
    SequenceId seq_id = 0;
    SequenceReservation reservation;
    SequenceScheduleCursor cursor;
    BlockTable block_table;
    int executed_frontier = 0;
    uint64_t parent_hash = 0;
    std::vector<int32_t> pending_hash_tokens;
};

// One canonical request object shared by waiting and running scheduling
// indexes. Execution state is materialized only after admission succeeds.
struct RequestState {
    std::string request_id;

    std::vector<int32_t> initial_prompt_tokens;
    std::vector<int32_t> prompt_tokens;
    RequestStatus status = RequestStatus::Active;
    std::vector<int32_t> generated_tokens;
    SamplingParams sampling;
    int max_context_len = 2048;

    TokenSink sink;
    // Materialized only after admission. All logical sequence state remains in
    // Scheduler; Worker receives only per-batch physical execution data.
    std::optional<SequenceSchedulingState> scheduling;
};

class EngineCore;
struct SchedulerTestAccess;

// Scheduler owns request policy and state. EngineCore owns the execution loop
// and batch completion FIFO; both run on the scheduler io_context thread.
class Scheduler {
public:
    Scheduler(asio::io_context& io, Executor& executor, EngineConfig config = {});
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Public methods post state changes to the scheduler io_context.
    void submit(SchedulerRequest req);
    void cancel(std::string request_id);
    void start();
    Capacity capacity() const;
    void shutdown();
    std::shared_future<void> shutdown_async();

private:
    friend class EngineCore;
    friend struct SchedulerTestAccess;

    using RequestPtr = std::shared_ptr<RequestState>;
    using RequestRegistry = std::unordered_map<std::string, RequestPtr>;
    using RunningOrder = std::list<RequestPtr>;
    using RunningIterator = RunningOrder::iterator;

    struct BatchBuildContext {
        ScheduledBatch batch;
        int token_budget = 0;
        int block_size = 0;
        bool sampling_set = false;
    };

    // All methods below run on scheduler_io unless explicitly marked public.
    void submit_on_scheduler_thread(SchedulerRequest req);
    void cancel_on_scheduler_thread(const std::string& request_id);
    void wake_on_scheduler_thread();
    void complete_shutdown_on_scheduler_thread();

    asio::awaitable<ScheduledBatch> schedule_step();
    asio::awaitable<bool> admit_one_skipped(BatchBuildContext& ctx);
    asio::awaitable<bool> admit_one_waiting(BatchBuildContext& ctx);
    asio::awaitable<bool> evict_one_skipped();
    asio::awaitable<bool> suspend_one_running();
    bool has_schedulable_work() const;
    bool has_waiting_work() const noexcept { return !waiting_.empty(); }
    bool admission_paused() const noexcept { return admission_paused_; }

    void build_running_batch(BatchBuildContext& ctx);
    void build_state_work(BatchBuildContext& ctx, RequestState& state);
    static void retire_reservation(RequestState& state, const WorkItem& item);
    static bool is_schedulable_state(const RequestState& state) noexcept;
    static bool is_prefill_phase(GenerationPhase phase) noexcept;
    static bool is_decode_phase(GenerationPhase phase) noexcept;
    static void prepare_for_wait(RequestState& state);
    void release_scheduling_blocks(RequestState& state);
    asio::awaitable<void> release_and_move_to_wait(const RequestPtr& request);

    asio::awaitable<void> update_from_output(const ScheduledBatch& batch,
                                             const BatchResult& result);
    void retire_work(const WorkItem& item, const WorkItemResult& result);
    void fail_batch(const ScheduledBatch& batch, ErrorCode err);
    asio::awaitable<void> cleanup_terminal_requests();
    asio::awaitable<void> cleanup_terminal_queue(std::deque<RequestPtr>& queue);
    asio::awaitable<void> preempt_one_for_admission();
    void fail_all_waiting(ErrorCode err);
    asio::awaitable<void> cleanup_all_running(ErrorCode shutdown_err);
    asio::awaitable<void> wait_for_work();

    bool send_event(const TokenSink& sink, Result<GeneratedToken> result);
    bool send_token_event(RequestState& state);
    bool send_terminal_event(RequestState& state);
    bool send_error_event(RequestState& state, ErrorCode err);
    void erase_request(const RequestPtr& request);

    asio::io_context& io_;
    Executor& executor_;
    EngineConfig engine_config_;
    asio::steady_timer idle_timer_;
    std::unique_ptr<EngineCore> core_;

    // requests_ is the canonical registry. A request has exactly one scheduling
    // position: waiting_ or running_. The indexes retain shared ownership of
    // that canonical state instead of copying it; by_seq_id_ is populated only
    // while the request is running.
    RequestRegistry requests_;
    std::deque<RequestPtr> waiting_;
    std::deque<RequestPtr> skip_;
    RunningOrder running_;
    std::unordered_map<SequenceId, RequestPtr> by_seq_id_;
    bool admission_paused_ = false;
    bool block_shortage_pending_ = false;

    std::atomic<bool> accepting_{false};
    uint64_t next_batch_id_{1};
    uint64_t next_seq_id_{1};
    BlockPool block_pool_;

    std::atomic<bool> shutdown_requested_{false};
    std::promise<void> shutdown_promise_;
    std::shared_future<void> shutdown_future_;
    bool shutdown_done_sent_{false};
};

}  // namespace ccinfer
