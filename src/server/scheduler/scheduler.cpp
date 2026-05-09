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

Scheduler::~Scheduler() { shutdown(); }

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void Scheduler::enqueue(std::shared_ptr<RequestState> req) {
    if (!req) return;
    if (!running_) {
        send_error_event(req, ErrorCode::ServerShuttingDown);
        return;
    }
    {
        std::lock_guard lock(incoming_mutex_);
        incoming_.push_back(std::move(req));
    }
    wake();
}

void Scheduler::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    asio::co_spawn(io_, do_work(), asio::detached);
}

void Scheduler::shutdown() {
    bool was_running = running_.exchange(false);
    if (!was_running) return;
    asio::post(io_, [this] { idle_timer_.cancel(); });
}

// ---------------------------------------------------------------------------
// Wake / idle
// ---------------------------------------------------------------------------

void Scheduler::wake() {
    asio::post(io_, [this] { idle_timer_.cancel(); });
}

asio::awaitable<void> Scheduler::wait_for_work() {
    if (!running_) co_return;
    {
        std::lock_guard lock(incoming_mutex_);
        if (!incoming_.empty()) co_return;
    }

    idle_timer_.expires_after(std::chrono::hours(24));
    auto [ec] = co_await idle_timer_.async_wait(as_tuple(deferred));
    (void)ec;  // operation_aborted on wake/shutdown — normal
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

asio::awaitable<void> Scheduler::do_work() {
    while (running_) {
        co_await drain_incoming();
        co_await cleanup_cancelled_or_finished();

        auto batch = build_scheduled_batch();

        if (batch.items.empty()) {
            co_await cleanup_cancelled_or_finished();
            {
                std::lock_guard lock(incoming_mutex_);
                if (!incoming_.empty()) continue;
            }
            if (!active_order_.empty()) continue;
            co_await wait_for_work();
            continue;
        }

        auto submitted_batch = batch;
        auto exec_result = co_await engine_.execute_batch(std::move(batch));

        if (!exec_result) {
            co_await fail_batch(submitted_batch, exec_result.error());
            co_await cleanup_cancelled_or_finished();
            continue;
        }

        co_await apply_and_push(submitted_batch, *exec_result);
        co_await cleanup_cancelled_or_finished();
    }

    fail_all_incoming();
    co_await cleanup_all_active(ErrorCode::ServerShuttingDown);
}

// ---------------------------------------------------------------------------
// Token event helpers
// ---------------------------------------------------------------------------

bool Scheduler::send_token_event(const std::shared_ptr<RequestState>& req) {
    GeneratedToken tok;
    tok.token_id = req->last_token;
    tok.has_token = true;
    tok.finished = req->finished;
    bool sent = req->output_channel->try_send(boost::system::error_code{}, tok);
    if (!sent) req->cancelled = true;
    return sent;
}

bool Scheduler::send_terminal_event(const std::shared_ptr<RequestState>& req) {
    req->finished = true;
    GeneratedToken tok;
    tok.has_token = false;
    tok.finished = true;
    bool sent = req->output_channel->try_send(boost::system::error_code{}, tok);
    if (!sent) req->cancelled = true;
    return sent;
}

bool Scheduler::send_error_event(const std::shared_ptr<RequestState>& req, ErrorCode err) {
    req->finished = true;
    GeneratedToken tok;
    tok.has_token = false;
    tok.finished = true;
    tok.error = err;
    bool sent = req->output_channel->try_send(boost::system::error_code{}, tok);
    if (!sent) req->cancelled = true;
    return sent;
}

// ---------------------------------------------------------------------------
// Drain incoming requests
// ---------------------------------------------------------------------------

asio::awaitable<void> Scheduler::drain_incoming() {
    std::vector<std::shared_ptr<RequestState>> batch;
    {
        std::lock_guard lock(incoming_mutex_);
        batch = std::move(incoming_);
        incoming_.clear();
    }

    for (auto& req : batch) {
        if (req->sampling.max_tokens <= 0) {
            send_terminal_event(req);
            continue;
        }

        auto result = co_await engine_.create_sequence(req->prompt_tokens, req->max_context_len);

        if (!result) {
            send_error_event(req, result.error());
            continue;
        }

        req->seq_id = *result;
        req->prefill_cursor = 0;
        req->prefill_done = false;
        req->last_token = -1;
        req->tokens_generated = 0;
        req->finished = false;
        req->cancelled = false;

        active_[*result] = req;
        active_order_.push_back(*result);
    }

    co_return;
}

// ---------------------------------------------------------------------------
// Build single-WorkItem batch (Phase 4.1)
// ---------------------------------------------------------------------------

ScheduledBatch Scheduler::build_scheduled_batch() {
    ScheduledBatch batch;
    batch.batch_id = next_batch_id_++;

    while (!active_order_.empty()) {
        SequenceId id = active_order_.front();
        auto it = active_.find(id);
        if (it == active_.end()) {
            active_order_.pop_front();
            continue;
        }

        auto& req = it->second;
        if (!req || req->finished || req->cancelled) {
            active_order_.pop_front();
            continue;
        }

        int prompt_len = static_cast<int>(req->prompt_tokens.size());
        if (req->prefill_cursor < 0) req->prefill_cursor = 0;
        if (req->prefill_cursor > prompt_len) {
            req->finished = true;
            send_error_event(req, ErrorCode::InvalidArgument);
            active_order_.pop_front();
            continue;
        }

        if (!req->prefill_done) {
            int remaining = prompt_len - req->prefill_cursor;
            if (remaining <= 0) {
                req->prefill_done = true;
                active_order_.pop_front();
                active_order_.push_back(id);
                continue;
            }
            batch.items.push_back(
                PrefillChunk{id, TokenSpan{req->prefill_cursor, remaining}, std::nullopt});
            active_order_.pop_front();
            active_order_.push_back(id);
            return batch;
        }

        // Decode
        if (req->tokens_generated >= req->sampling.max_tokens) {
            req->finished = true;
            active_order_.pop_front();
            continue;
        }

        if (req->last_token < 0) {
            req->finished = true;
            send_error_event(req, ErrorCode::BatchTranslationFailed);
            active_order_.pop_front();
            continue;
        }

        batch.items.push_back(DecodeOneToken{id, req->last_token, std::nullopt});
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
    // Batch-level consistency: batch_id and item count must match.
    if (result.batch_id != batch.batch_id || result.items.size() != batch.items.size()) {
        co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
        co_return;
    }

    for (const auto& wr : result.items) {
        // If item_index is out of range, the whole result is untrusted.
        if (wr.item_index < 0 || static_cast<size_t>(wr.item_index) >= batch.items.size()) {
            co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
            co_return;
        }

        // Validate seq_id and kind match the original WorkItem.
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
        auto& req = it->second;
        if (!req || req->cancelled) continue;

        bool has_new_token = false;

        if (wr.kind == WorkKind::PrefillChunk) {
            int prompt_len = static_cast<int>(req->prompt_tokens.size());
            int remaining = prompt_len - req->prefill_cursor;

            if (remaining <= 0 || wr.tokens_consumed <= 0 || wr.tokens_consumed > remaining) {
                send_error_event(req, ErrorCode::BatchTranslationFailed);
                continue;
            }

            req->prefill_cursor += wr.tokens_consumed;
            req->prefill_done = (req->prefill_cursor >= prompt_len);

            if (!wr.sampled_tokens.empty()) {
                // Phase 4.1: consume only the first sampled token.
                // TODO: support multiple sampled tokens if runner returns more than one.
                req->last_token = wr.sampled_tokens[0];
                req->tokens_generated++;
                has_new_token = true;
            }
        } else {
            // DecodeOneToken: must consume exactly one token.
            if (wr.tokens_consumed != 1) {
                send_error_event(req, ErrorCode::BatchTranslationFailed);
                continue;
            }

            if (wr.sampled_tokens.empty() && !wr.eos) {
                send_error_event(req, ErrorCode::BatchTranslationFailed);
                continue;
            }

            if (!wr.sampled_tokens.empty()) {
                // Phase 4.1: consume only the first sampled token.
                // TODO: support multiple sampled tokens if runner returns more than one.
                req->last_token = wr.sampled_tokens[0];
                req->tokens_generated++;
                has_new_token = true;
            }
        }

        // Unified finished check
        if (wr.eos) req->finished = true;
        if (req->sampling.max_tokens <= 0) req->finished = true;
        if (req->tokens_generated >= req->sampling.max_tokens) req->finished = true;

        // Send event
        if (has_new_token) {
            send_token_event(req);
        } else if (req->finished) {
            send_terminal_event(req);
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
        auto& req = it->second;
        if (!req) continue;
        if (req->finished || req->cancelled) continue;
        new_order.push_back(id);
    }
    active_order_ = std::move(new_order);
}

asio::awaitable<void> Scheduler::cleanup_cancelled_or_finished() {
    std::vector<SequenceId> to_release;
    for (const auto& [seq_id, req] : active_) {
        if (req && (req->cancelled || req->finished)) {
            to_release.push_back(seq_id);
        }
    }

    for (SequenceId seq_id : to_release) {
        auto r = co_await engine_.release_sequence(seq_id);
        if (!r) { /* TODO: log release failure */
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

        auto& req = it->second;
        if (!req) continue;

        send_error_event(req, err);
    }
    // Release happens in cleanup_cancelled_or_finished() after return.
    co_return;
}

void Scheduler::fail_all_incoming() {
    std::vector<std::shared_ptr<RequestState>> remaining;
    {
        std::lock_guard lock(incoming_mutex_);
        remaining = std::move(incoming_);
        incoming_.clear();
    }
    for (auto& req : remaining) {
        send_error_event(req, ErrorCode::ServerShuttingDown);
    }
}

asio::awaitable<void> Scheduler::cleanup_all_active(ErrorCode shutdown_err) {
    std::vector<SequenceId> seqs;
    for (auto& [seq_id, req] : active_) {
        seqs.push_back(seq_id);
    }

    for (SequenceId seq_id : seqs) {
        auto it = active_.find(seq_id);
        if (it == active_.end()) continue;

        auto& req = it->second;
        if (req && !req->finished && !req->cancelled) {
            send_error_event(req, shutdown_err);
        }

        auto r = co_await engine_.release_sequence(seq_id);
        if (!r) { /* TODO: log */
        }

        active_.erase(seq_id);
    }

    active_order_.clear();
    co_return;
}

}  // namespace server
}  // namespace ccinfer
