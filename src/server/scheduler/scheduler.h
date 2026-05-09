#pragma once

#include <atomic>
#include <boost/asio/io_context.hpp>
#include <boost/asio/steady_timer.hpp>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "engine/engine.h"
#include "server/common/types.h"

namespace ccinfer {
namespace server {

namespace asio = boost::asio;
using engine::Engine;

// Scheduler must outlive the detached do_work() coroutine.
// Server shutdown must call shutdown() and keep io_context running
// until do_work() exits.

class Scheduler {
public:
    Scheduler(asio::io_context& io, Engine& engine);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void enqueue(std::shared_ptr<RequestState> req);
    void start();
    void shutdown();

private:
    using SeqMap = std::unordered_map<SequenceId, std::shared_ptr<RequestState>>;

    asio::awaitable<void> do_work();
    asio::awaitable<void> drain_incoming();
    ScheduledBatch build_scheduled_batch();
    asio::awaitable<void> apply_and_push(const ScheduledBatch& batch, const BatchResult& result);
    asio::awaitable<void> cleanup_cancelled_or_finished();
    asio::awaitable<void> fail_batch(const ScheduledBatch& batch, ErrorCode err);
    void fail_all_incoming();
    asio::awaitable<void> cleanup_all_active(ErrorCode shutdown_err);
    asio::awaitable<void> wait_for_work();

    void filter_active_order();
    void wake();

    bool send_token_event(const std::shared_ptr<RequestState>& req);
    bool send_terminal_event(const std::shared_ptr<RequestState>& req);
    bool send_error_event(const std::shared_ptr<RequestState>& req, ErrorCode err);

    asio::io_context& io_;
    Engine& engine_;
    asio::steady_timer idle_timer_;

    std::vector<std::shared_ptr<RequestState>> incoming_;
    std::mutex incoming_mutex_;

    SeqMap active_;
    std::deque<SequenceId> active_order_;

    std::atomic<bool> running_{false};
    uint64_t next_batch_id_{1};
};

}  // namespace server
}  // namespace ccinfer
