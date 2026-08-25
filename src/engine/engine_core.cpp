#include "engine/engine_core.h"

#include <utility>

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/detached.hpp>

#include "executor/executor.h"
#include "scheduler/scheduler.h"

namespace ccinfer {

namespace asio = boost::asio;

EngineCore::EngineCore(asio::io_context& io, Scheduler& scheduler, Executor& executor,
                       EngineConfig config)
    : io_(io), scheduler_(scheduler), executor_(executor), config_(config) {}

void EngineCore::start() { asio::co_spawn(io_, run(), asio::detached); }

void EngineCore::request_shutdown() {
    scheduler_.fail_all_waiting(ErrorCode::ServerShuttingDown);
    scheduler_.wake_on_scheduler_thread();
}

asio::awaitable<void> EngineCore::run() {
    while (scheduler_.accepting_.load()) {
        while (in_flight_.size() < static_cast<std::size_t>(config_.max_concurrent_batches)) {
            auto batch = co_await scheduler_.schedule_step();
            if (batch.items.empty()) break;

            auto future_r = executor_.execute_batch(batch);
            if (!future_r) {
                if (future_r.error() == ErrorCode::KVBlockExhausted) {
                    co_await scheduler_.handle_batch_error(batch, future_r.error());
                } else {
                    scheduler_.mark_dispatch_failed(batch, future_r.error());
                }
                co_await scheduler_.cleanup_terminal_requests();
                continue;
            }

            in_flight_.push_back(InFlightBatch{std::move(batch), std::move(*future_r)});
        }

        if (in_flight_.empty()) {
            co_await scheduler_.cleanup_terminal_requests();
            if (scheduler_.has_waiting_work() || scheduler_.has_schedulable_work()) {
                co_await scheduler_.preempt_one_for_admission();
                continue;
            }
            co_await scheduler_.wait_for_work();
            continue;
        }

        co_await collect_oldest();
        co_await scheduler_.cleanup_terminal_requests();
    }

    scheduler_.fail_all_waiting(ErrorCode::ServerShuttingDown);
    co_await drain_in_flight();
    co_await scheduler_.cleanup_all_running(ErrorCode::ServerShuttingDown);
    scheduler_.complete_shutdown_on_scheduler_thread();
}

asio::awaitable<void> EngineCore::collect_oldest() {
    if (in_flight_.empty()) co_return;

    auto current = std::move(in_flight_.front());
    in_flight_.pop_front();

    auto result = co_await executor_.collect_batch(std::move(current.future));
    if (result) {
        co_await scheduler_.update_from_output(current.batch, *result);
    } else {
        co_await scheduler_.handle_batch_error(current.batch, result.error());
    }
}

asio::awaitable<void> EngineCore::drain_in_flight() {
    while (!in_flight_.empty()) co_await collect_oldest();
}

}  // namespace ccinfer
