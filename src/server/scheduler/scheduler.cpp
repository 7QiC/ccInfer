#include "server/scheduler/scheduler.h"

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/post.hpp>
#include <chrono>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

namespace ccinfer {
namespace server {

namespace {
using asio::as_tuple;
using asio::deferred;
}  // namespace

Scheduler::Scheduler(asio::io_context& io, Engine& engine)
    : io_(io), engine_(engine), idle_timer_(io) {}

// PRECONDITION: shutdown_async().wait() must have completed before destruction.
// The detached do_work() coroutine captures `this`; destroying the Scheduler
// while do_work() is still running is use-after-free.
// shutdown_async().wait() must NOT be called on the scheduler io_context thread
// (would deadlock since do_work() runs on that thread and wait() blocks it).
Scheduler::~Scheduler() = default;

// ---------------------------------------------------------------------------
// Thread-safe public API
// ---------------------------------------------------------------------------

void Scheduler::submit(SchedulerRequest req) {
    if (!running_) {
        send_event(req.sink, std::unexpected(ErrorCode::ServerShuttingDown));
        return;
    }
    asio::post(io_, [this, req = std::move(req)]() mutable {
        if (!running_) {
            send_event(req.sink, std::unexpected(ErrorCode::ServerShuttingDown));
            return;
        }
        submit_on_scheduler_thread(std::move(req));
    });
}

void Scheduler::cancel(std::string request_id) {
    // Safe no-op during/after shutdown — the lambda checks internal state.
    asio::post(io_, [this, request_id = std::move(request_id)] {
        cancel_on_scheduler_thread(request_id);
        wake_on_scheduler_thread();
    });
}

void Scheduler::start() {
    // Phase 4.1: one-shot.  After shutdown, start() is a no-op to prevent
    // accidentally re-launching do_work() on a partially-destroyed object.
    std::lock_guard lock(shutdown_mutex_);
    if (shutdown_promise_) return;
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    asio::co_spawn(io_, do_work(), asio::detached);
}

void Scheduler::shutdown() { shutdown_async(); }

std::shared_future<void> Scheduler::shutdown_async() {
    std::lock_guard lock(shutdown_mutex_);

    if (shutdown_promise_) {
        return shutdown_future_;
    }

    shutdown_promise_ = std::make_unique<std::promise<void>>();
    shutdown_future_ = shutdown_promise_->get_future().share();

    bool was_running = running_.exchange(false);

    asio::post(io_, [this, was_running] {
        fail_all_pending_on_scheduler_thread(ErrorCode::ServerShuttingDown);
        idle_timer_.cancel();

        // If do_work() was never started, complete immediately.
        if (!was_running) {
            complete_shutdown_on_scheduler_thread();
        }
        // Otherwise do_work() will call complete_shutdown_on_scheduler_thread()
        // at its tail after cleanup_all_active.
    });

    return shutdown_future_;
}

// ---------------------------------------------------------------------------
// Scheduler-thread-only helpers
// ---------------------------------------------------------------------------

void Scheduler::submit_on_scheduler_thread(SchedulerRequest req) {
    if (pending_or_creating_ids_.count(req.request_id) || by_request_id_.count(req.request_id)) {
        send_event(req.sink, std::unexpected(ErrorCode::InvalidArgument));
        return;
    }

    pending_or_creating_ids_.insert(req.request_id);
    pending_requests_.push_back(std::move(req));
    wake_on_scheduler_thread();
}

void Scheduler::cancel_on_scheduler_thread(const std::string& request_id) {
    auto it = by_request_id_.find(request_id);
    if (it != by_request_id_.end()) {
        it->second->cancelled = true;
        return;
    }

    if (pending_or_creating_ids_.count(request_id)) {
        cancelled_pending_ids_.insert(request_id);
        return;
    }
}

void Scheduler::wake_on_scheduler_thread() { idle_timer_.cancel(); }

void Scheduler::complete_shutdown_on_scheduler_thread() {
    if (shutdown_done_sent_) return;
    shutdown_done_sent_ = true;
    if (shutdown_promise_) {
        shutdown_promise_->set_value();
    }
}

// ---------------------------------------------------------------------------
// send_event — post a token/event to the HTTP io_context via TokenSink
// ---------------------------------------------------------------------------

// send_event returns false only when sink.channel has already expired
// (synchronous failure).  try_send failures (buffer full) are asynchronous
// and converge via on_send_failed → cancel().  Callers that set
// state->cancelled based on the return value therefore only handle the
// immediate-expired case; the buffer-full path is handled by the cancel
// callback.
//
// Phase 4.1: try_send buffer-full → treated as cancelled request.
// Future: async_send / HTTP-side queue for real back-pressure.
bool Scheduler::send_event(const TokenSink& sink, Result<GeneratedToken> result) {
    if (sink.channel.expired()) {
        return false;
    }

    asio::post(sink.executor, [weak = sink.channel, result = std::move(result),
                               on_send_failed = sink.on_send_failed]() mutable {
        auto ch = weak.lock();
        if (!ch) {
            if (on_send_failed) on_send_failed();
            return;
        }
        bool ok = ch->try_send(boost::system::error_code{}, std::move(result));
        if (!ok && on_send_failed) {
            on_send_failed();
        }
    });

    return true;
}

// Terminal-event contract:
//   When the last token reaches max_tokens (or EOS in future), the scheduler
//   sends a token event with has_token=true + finished=true.  No separate
//   terminal event (has_token=false, finished=true) follows.
//   The HTTP/SSE consumer MUST end the stream when finished=true on any event,
//   NOT only on a has_token=false terminal event.
bool Scheduler::send_token_event(StatePtr& state) {
    GeneratedToken tok;
    tok.token_id = state->last_token;
    tok.has_token = true;
    tok.finished = state->finished;
    bool sent = send_event(state->sink, std::move(tok));
    if (!sent) state->cancelled = true;
    return sent;
}

bool Scheduler::send_terminal_event(StatePtr& state) {
    state->finished = true;
    GeneratedToken tok{.has_token = false, .finished = true};
    bool sent = send_event(state->sink, std::move(tok));
    if (!sent) state->cancelled = true;
    return sent;
}

bool Scheduler::send_error_event(StatePtr& state, ErrorCode err) {
    state->finished = true;
    bool sent = send_event(state->sink, std::unexpected(err));
    if (!sent) state->cancelled = true;
    return sent;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

asio::awaitable<void> Scheduler::do_work() {
    while (running_) {
        co_await drain_pending();
        if (!running_) break;

        co_await cleanup_cancelled_or_finished();
        if (!running_) break;

        auto batch = build_scheduled_batch();

        if (batch.items.empty()) {
            co_await cleanup_cancelled_or_finished();
            if (!pending_requests_.empty()) continue;
            if (!active_order_.empty()) continue;
            co_await wait_for_work();
            continue;
        }

        auto submitted_batch = batch;
        auto exec_result = co_await engine_.execute_batch(std::move(batch));
        if (!running_) break;

        if (!exec_result) {
            co_await fail_batch(submitted_batch, exec_result.error());
            co_await cleanup_cancelled_or_finished();
            continue;
        }

        co_await apply_and_push(submitted_batch, *exec_result);
        co_await cleanup_cancelled_or_finished();
    }

    fail_all_pending_on_scheduler_thread(ErrorCode::ServerShuttingDown);
    co_await cleanup_all_active(ErrorCode::ServerShuttingDown);

    complete_shutdown_on_scheduler_thread();
}

// ---------------------------------------------------------------------------
// Drain pending requests — create sequences, promote to active
// ---------------------------------------------------------------------------

asio::awaitable<void> Scheduler::drain_pending() {
    auto requests = std::move(pending_requests_);
    pending_requests_.clear();

    for (auto& req : requests) {
        // Shutdown may have occurred — stop processing remaining local requests.
        if (!running_) {
            pending_or_creating_ids_.erase(req.request_id);
            send_event(req.sink, std::unexpected(ErrorCode::ServerShuttingDown));
            continue;
        }

        // Check if cancelled while pending (before create_sequence).
        if (cancelled_pending_ids_.erase(req.request_id) > 0) {
            pending_or_creating_ids_.erase(req.request_id);
            send_event(req.sink, std::unexpected(ErrorCode::RequestCancelled));
            continue;
        }

        if (req.sampling.max_tokens <= 0) {
            pending_or_creating_ids_.erase(req.request_id);
            GeneratedToken tok{.has_token = false, .finished = true};
            send_event(req.sink, std::move(tok));
            continue;
        }

        auto result = co_await engine_.create_sequence(req.prompt_tokens, req.max_context_len);

        // Shutdown may have occurred during the await — don't promote to active.
        if (!running_) {
            pending_or_creating_ids_.erase(req.request_id);

            if (result) {
                auto r = co_await engine_.release_sequence(*result);
                if (!r) { /* TODO: log release failure */
                }
            }

            send_event(req.sink, std::unexpected(ErrorCode::ServerShuttingDown));
            continue;
        }

        // Check if cancelled during the create_sequence await.
        if (cancelled_pending_ids_.erase(req.request_id) > 0) {
            pending_or_creating_ids_.erase(req.request_id);

            if (result) {
                auto r = co_await engine_.release_sequence(*result);
                if (!r) { /* TODO: log release failure */
                }
            }

            send_event(req.sink, std::unexpected(ErrorCode::RequestCancelled));
            continue;
        }

        if (!result) {
            pending_or_creating_ids_.erase(req.request_id);
            send_event(req.sink, std::unexpected(result.error()));
            continue;
        }

        pending_or_creating_ids_.erase(req.request_id);

        auto state = std::make_shared<SchedulerRequestState>();
        state->request_id = std::move(req.request_id);
        state->seq_id = *result;
        state->prompt_tokens = std::move(req.prompt_tokens);
        state->sampling = req.sampling;
        state->max_context_len = req.max_context_len;
        state->sink = std::move(req.sink);

        by_request_id_[state->request_id] = state;
        seq_to_request_id_[*result] = state->request_id;
        active_[*result] = state;
        active_order_.push_back(*result);
    }

    co_return;
}

// ---------------------------------------------------------------------------
// Wait for work (idle)
// ---------------------------------------------------------------------------

asio::awaitable<void> Scheduler::wait_for_work() {
    if (!running_) co_return;
    if (!pending_requests_.empty()) co_return;

    idle_timer_.expires_after(std::chrono::hours(24));
    auto [ec] = co_await idle_timer_.async_wait(as_tuple(deferred));
    (void)ec;
}

// ---------------------------------------------------------------------------
// Build single-WorkItem batch (Phase 4.1)
// ---------------------------------------------------------------------------

ScheduledBatch Scheduler::build_scheduled_batch() {
    ScheduledBatch batch;
    batch.batch_id = next_batch_id_++;
    // Phase 4.1: copy sampling from the first active request.
    // Future: per-item sampling via std::vector<SamplingParams>.

    while (!active_order_.empty()) {
        SequenceId id = active_order_.front();
        auto it = active_.find(id);
        if (it == active_.end()) {
            active_order_.pop_front();
            continue;
        }

        auto& state = it->second;
        if (!state || state->finished || state->cancelled) {
            active_order_.pop_front();
            continue;
        }

        int prompt_len = static_cast<int>(state->prompt_tokens.size());
        if (state->prefill_cursor < 0) state->prefill_cursor = 0;
        if (state->prefill_cursor > prompt_len) {
            state->finished = true;
            send_error_event(state, ErrorCode::InvalidArgument);
            active_order_.pop_front();
            continue;
        }

        if (!state->prefill_done) {
            int remaining = prompt_len - state->prefill_cursor;
            if (remaining <= 0) {
                state->prefill_done = true;
                active_order_.pop_front();
                active_order_.push_back(id);
                continue;
            }
            batch.sampling = state->sampling;
            batch.items.push_back(
                PrefillChunk{id, TokenSpan{state->prefill_cursor, remaining}, std::nullopt});
            active_order_.pop_front();
            active_order_.push_back(id);
            return batch;
        }

        // Decode
        if (state->tokens_generated >= state->sampling.max_tokens) {
            state->finished = true;
            active_order_.pop_front();
            continue;
        }

        if (state->last_token < 0) {
            state->finished = true;
            send_error_event(state, ErrorCode::BatchTranslationFailed);
            active_order_.pop_front();
            continue;
        }

        batch.sampling = state->sampling;
        batch.items.push_back(DecodeOneToken{id, state->last_token, std::nullopt});
        active_order_.pop_front();
        active_order_.push_back(id);
        return batch;
    }

    return batch;
}

// ---------------------------------------------------------------------------
// Apply batch result
// ---------------------------------------------------------------------------

asio::awaitable<void> Scheduler::apply_and_push(const ScheduledBatch& batch,
                                                const BatchResult& result) {
    if (result.batch_id != batch.batch_id || result.items.size() != batch.items.size()) {
        co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
        co_return;
    }

    for (const auto& wr : result.items) {
        if (wr.item_index < 0 || static_cast<size_t>(wr.item_index) >= batch.items.size()) {
            co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
            co_return;
        }

        const auto& original_item = batch.items[wr.item_index];
        SequenceId expected_seq_id = 0;
        WorkKind expected_kind = WorkKind::PrefillChunk;
        std::visit(
            [&](const auto& w) {
                expected_seq_id = w.seq_id;
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    expected_kind = WorkKind::PrefillChunk;
                } else {
                    expected_kind = WorkKind::DecodeOneToken;
                }
            },
            original_item);

        if (expected_seq_id != wr.seq_id || wr.kind != expected_kind) {
            co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
            co_return;
        }

        auto it = active_.find(wr.seq_id);
        if (it == active_.end()) continue;
        auto& state = it->second;
        if (!state || state->cancelled) continue;

        bool has_new_token = false;

        if (wr.kind == WorkKind::PrefillChunk) {
            int prompt_len = static_cast<int>(state->prompt_tokens.size());
            int remaining = prompt_len - state->prefill_cursor;

            if (remaining <= 0 || wr.tokens_consumed <= 0 || wr.tokens_consumed > remaining) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }

            state->prefill_cursor += wr.tokens_consumed;
            state->prefill_done = (state->prefill_cursor >= prompt_len);

            if (!wr.sampled_tokens.empty()) {
                state->last_token = wr.sampled_tokens[0];
                state->tokens_generated++;
                has_new_token = true;
            }
        } else {
            if (wr.tokens_consumed != 1) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }

            if (wr.sampled_tokens.empty() && !wr.eos) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }

            if (!wr.sampled_tokens.empty()) {
                state->last_token = wr.sampled_tokens[0];
                state->tokens_generated++;
                has_new_token = true;
            }
        }

        if (wr.eos) state->finished = true;
        if (state->sampling.max_tokens <= 0) state->finished = true;
        if (state->tokens_generated >= state->sampling.max_tokens) state->finished = true;

        if (has_new_token) {
            send_token_event(state);
        } else if (state->finished) {
            send_terminal_event(state);
        }
    }

    co_return;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void Scheduler::filter_active_order() {
    std::deque<SequenceId> new_order;
    for (SequenceId id : active_order_) {
        auto it = active_.find(id);
        if (it == active_.end()) continue;
        auto& state = it->second;
        if (!state) continue;
        if (state->finished || state->cancelled) continue;
        new_order.push_back(id);
    }
    active_order_ = std::move(new_order);
}

asio::awaitable<void> Scheduler::cleanup_cancelled_or_finished() {
    std::vector<SequenceId> to_release;
    for (const auto& [seq_id, state] : active_) {
        if (state && (state->cancelled || state->finished)) {
            to_release.push_back(seq_id);
        }
    }

    for (SequenceId seq_id : to_release) {
        auto r = co_await engine_.release_sequence(seq_id);
        if (!r) { /* TODO: log release failure */
        }

        auto sit = seq_to_request_id_.find(seq_id);
        if (sit != seq_to_request_id_.end()) {
            by_request_id_.erase(sit->second);
            seq_to_request_id_.erase(sit);
        }
        active_.erase(seq_id);
    }

    filter_active_order();
    co_return;
}

asio::awaitable<void> Scheduler::fail_batch(const ScheduledBatch& batch, ErrorCode err) {
    std::unordered_set<SequenceId> seen;
    for (const auto& item : batch.items) {
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        if (!seen.insert(seq_id).second) continue;

        auto it = active_.find(seq_id);
        if (it == active_.end()) continue;

        auto& state = it->second;
        if (!state) continue;

        send_error_event(state, err);
    }
    co_return;
}

void Scheduler::fail_all_pending_on_scheduler_thread(ErrorCode err) {
    auto requests = std::move(pending_requests_);
    pending_requests_.clear();
    pending_or_creating_ids_.clear();
    cancelled_pending_ids_.clear();

    for (auto& req : requests) {
        send_event(req.sink, std::unexpected(err));
    }
}

asio::awaitable<void> Scheduler::cleanup_all_active(ErrorCode shutdown_err) {
    std::vector<SequenceId> seqs;
    for (const auto& [seq_id, _] : active_) {
        seqs.push_back(seq_id);
    }

    for (SequenceId seq_id : seqs) {
        auto it = active_.find(seq_id);
        if (it == active_.end()) continue;

        auto& state = it->second;
        if (state && !state->finished && !state->cancelled) {
            send_error_event(state, shutdown_err);
        }

        auto r = co_await engine_.release_sequence(seq_id);
        if (!r) { /* TODO: log */
        }

        active_.erase(seq_id);

        auto sit = seq_to_request_id_.find(seq_id);
        if (sit != seq_to_request_id_.end()) {
            by_request_id_.erase(sit->second);
            seq_to_request_id_.erase(sit);
        }
    }

    active_order_.clear();
    by_request_id_.clear();
    seq_to_request_id_.clear();
    co_return;
}

}  // namespace server
}  // namespace ccinfer
