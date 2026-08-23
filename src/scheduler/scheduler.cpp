#include "scheduler/scheduler.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

#include <boost/asio/as_tuple.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/deferred.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/post.hpp>

#include "facade/log.h"

namespace ccinfer {

namespace {
using asio::as_tuple;
using asio::deferred;

bool is_prefill_phase(GenerationPhase phase) noexcept {
    return phase == GenerationPhase::Prefill || phase == GenerationPhase::ReplayPrefill;
}

bool is_decode_phase(GenerationPhase phase) noexcept {
    return phase == GenerationPhase::Bootstrap || phase == GenerationPhase::Decode;
}

}  // namespace

Scheduler::Scheduler(asio::io_context& io, Executor& executor, EngineConfig config)
    : io_(io), executor_(executor), engine_config_(config), idle_timer_(io) {}

// PRECONDITION: shutdown_async().wait() must have completed before destruction.
// The detached do_work() coroutine captures `this`; destroying the Scheduler
// while do_work() is still running is use-after-free.
// shutdown_async().wait() must NOT be called on the scheduler io_context thread
// (would deadlock since do_work() runs on that thread and wait() blocks it).
Scheduler::~Scheduler() = default;

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
    asio::post(io_, [this, request_id = std::move(request_id)] {
        cancel_on_scheduler_thread(request_id);
        wake_on_scheduler_thread();
    });
}

Capacity Scheduler::capacity() const { return executor_.capacity(); }

void Scheduler::start() {
    // One-shot: after shutdown, start() is a no-op to prevent re-launching
    // do_work() on a partially-destroyed object.
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

void Scheduler::submit_on_scheduler_thread(SchedulerRequest req) {
    if (pending_or_creating_ids_.count(req.request_id) || by_request_id_.count(req.request_id)) {
        send_event(req.sink, std::unexpected(ErrorCode::InvalidArgument));
        return;
    }

    if (req.max_context_len <= 0) req.max_context_len = engine_config_.default_max_context_len;
    pending_or_creating_ids_.insert(req.request_id);
    pending_requests_.push_back(std::move(req));
    wake_on_scheduler_thread();
}

void Scheduler::cancel_on_scheduler_thread(const std::string& request_id) {
    auto it = by_request_id_.find(request_id);
    if (it != by_request_id_.end()) {
        if (it->second->status == RequestStatus::Active) {
            it->second->status = RequestStatus::Cancelled;
        }
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

// send_event returns false only when sink.channel has already expired
// synchronously.  A full channel is normal HTTP/SSE back-pressure, not a
// request failure: async_send waits until the HTTP coroutine receives space.
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
        ch->async_send(boost::system::error_code{}, std::move(result),
                       [on_send_failed](boost::system::error_code ec) {
                           if (ec && on_send_failed) on_send_failed();
                       });
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
    tok.finished = state->status == RequestStatus::Finished;
    bool sent = send_event(state->sink, std::move(tok));
    if (!sent) {
        state->sink_disconnected = true;
        if (state->status == RequestStatus::Active) state->status = RequestStatus::Cancelled;
    }
    return sent;
}

bool Scheduler::send_terminal_event(StatePtr& state) {
    state->status = RequestStatus::Finished;
    GeneratedToken tok{.has_token = false, .finished = true};
    bool sent = send_event(state->sink, std::move(tok));
    if (!sent) state->sink_disconnected = true;
    return sent;
}

bool Scheduler::send_error_event(StatePtr& state, ErrorCode err) {
    state->status = RequestStatus::Failed;
    bool sent = send_event(state->sink, std::unexpected(err));
    if (!sent) state->sink_disconnected = true;
    return sent;
}

asio::awaitable<void> Scheduler::do_work() {
    while (running_) {
        co_await drain_pending();
        if (!running_) break;

        co_await cleanup_terminal_requests();
        if (!running_) break;

        auto batch = build_scheduled_batch();

        if (batch.items.empty()) {
            auto active_before = active_.size();
            co_await cleanup_terminal_requests();
            if (!pending_requests_.empty()) continue;

            if (!active_order_.empty()) {
                // Cleanup may have released KV blocks — retry if so.
                if (active_.size() < active_before) continue;

                // Budget stalled: active items exist but none schedulable.
                // Suspend budget-blocked items as the last-resort KV pressure
                // valve. Suspended requests stay active and are replay-prefilled
                // when capacity is available again.
                if (!budget_blocked_.empty()) {
                    for (SequenceId id : budget_blocked_) {
                        auto it = active_.find(id);
                        if (it != active_.end() && is_schedulable_state(it->second)) {
                            co_await suspend_sequence_for_replay(it->second);
                        }
                    }
                    budget_blocked_.clear();
                } else {
                    // Unlikely: active_ non-empty but nothing blocked by budget
                    // (e.g. all sampling-incompatible). Short idle to avoid spin.
                    idle_timer_.expires_after(std::chrono::milliseconds(100));
                    auto [ec] = co_await idle_timer_.async_wait(as_tuple(deferred));
                    (void)ec;
                }
                continue;
            }
            co_await wait_for_work();
            continue;
        }

        auto submitted_batch = batch;
        auto exec_result = co_await executor_.execute_batch(std::move(batch));
        if (!running_) break;

        if (!exec_result) {
            if (exec_result.error() == ErrorCode::KVBlockExhausted) {
                std::unordered_set<SequenceId> seen;
                for (const auto& item : submitted_batch.items) {
                    SequenceId seq_id = 0;
                    std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
                    if (!seen.insert(seq_id).second) continue;

                    auto it = active_.find(seq_id);
                    if (it != active_.end() && is_schedulable_state(it->second)) {
                        co_await suspend_sequence_for_replay(it->second);
                    }
                }
            } else {
                co_await fail_batch(submitted_batch, exec_result.error());
            }
            co_await cleanup_terminal_requests();
            continue;
        }

        co_await apply_and_push(submitted_batch, *exec_result);
        co_await cleanup_terminal_requests();
    }

    fail_all_pending_on_scheduler_thread(ErrorCode::ServerShuttingDown);
    co_await cleanup_all_active(ErrorCode::ServerShuttingDown);

    complete_shutdown_on_scheduler_thread();
}

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

        auto result = co_await executor_.create_sequence(req.prompt_tokens, req.max_context_len);

        // Shutdown may have occurred during the await — don't promote to active.
        if (!running_) {
            pending_or_creating_ids_.erase(req.request_id);

            if (result) {
                auto r = co_await executor_.release_sequence(result->seq_id);
                if (!r) {
                    ccLog::warn("release_sequence failed seq={} err={}", result->seq_id,
                                static_cast<int>(r.error()));
                }
            }

            send_event(req.sink, std::unexpected(ErrorCode::ServerShuttingDown));
            continue;
        }

        // Check if cancelled during the create_sequence await.
        if (cancelled_pending_ids_.erase(req.request_id) > 0) {
            pending_or_creating_ids_.erase(req.request_id);

            if (result) {
                auto r = co_await executor_.abort_sequence(result->seq_id);
                if (!r) {
                    ccLog::warn("abort_sequence failed seq={} err={}", result->seq_id,
                                static_cast<int>(r.error()));
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
        state->seq_id = result->seq_id;
        state->prompt_tokens = std::move(req.prompt_tokens);
        state->initial_prompt_tokens = state->prompt_tokens;
        const int prompt_len = static_cast<int>(state->prompt_tokens.size());
        if (result->prompt_processed > 0 && result->prompt_processed < prompt_len) {
            state->prefill_cursor = result->prompt_processed;
        } else if (result->prompt_processed >= prompt_len) {
            state->prefill_cursor = prompt_len;
            state->phase = GenerationPhase::Bootstrap;
        }
        state->sampling = req.sampling;
        state->max_context_len = req.max_context_len;
        state->sink = std::move(req.sink);

        by_request_id_[state->request_id] = state;
        seq_to_request_id_[result->seq_id] = state->request_id;
        active_[result->seq_id] = state;
        active_order_.push_back(result->seq_id);
    }

    co_return;
}

asio::awaitable<void> Scheduler::wait_for_work() {
    if (!running_) co_return;
    if (!pending_requests_.empty()) co_return;

    idle_timer_.expires_after(std::chrono::hours(24));
    auto [ec] = co_await idle_timer_.async_wait(as_tuple(deferred));
    (void)ec;
}

// Mixed scheduling uses one compute budget per engine step. Decode items cost
// one budget unit each; prefill items cost their chunk token count. Existing
// decode and in-progress prefill work can continue, while new prefill admission
// is capped to avoid overfilling KV with too many live sequences.

ScheduledBatch Scheduler::build_scheduled_batch() {
    BatchBuildContext ctx;
    ctx.batch.batch_id = next_batch_id_++;
    budget_blocked_.clear();

    auto cap = executor_.capacity();
    ctx.block_size = cap.block_size;
    if (ctx.block_size <= 0) {
        // Executor not initialised — fail all active to avoid busy spin.
        for (auto& [id, st] : active_) {
            if (is_schedulable_state(st)) send_error_event(st, ErrorCode::InternalError);
        }
        return ctx.batch;
    }
    // KVCacheManager can satisfy new block allocations by evicting cached-idle
    // prefix blocks, not only by consuming currently free blocks. Keep the
    // scheduler's admission budget aligned with allocator behavior.
    ctx.budget = cap.free_blocks + cap.block_cached_idle;
    ctx.compute_budget = engine_config_.schedule_compute_budget;

    for (const auto& [_, state] : active_) {
        if (is_schedulable_state(state) && has_started_execution(state)) ++ctx.started_active;
    }

    auto has_prefill_work = [&]() {
        for (SequenceId id : active_order_) {
            auto it = active_.find(id);
            if (it == active_.end()) continue;
            auto& state = it->second;
            if (!is_schedulable_state(state) || !is_prefill_phase(state->phase)) continue;
            int prompt_len = static_cast<int>(state->prompt_tokens.size());
            int cursor = std::max(state->prefill_cursor, 0);
            if (cursor < prompt_len) return true;
        }
        return false;
    };

    const bool prefill_available = has_prefill_work();
    build_decode_batch(ctx);
    if (prefill_available && ctx.compute_budget > 0) {
        build_prefill_batch(ctx);
    }

    if (!ctx.batch.items.empty()) {
        reorder_after_batch(ctx.included);
    }

    return ctx.batch;
}

bool Scheduler::is_schedulable_state(const StatePtr& state) noexcept {
    return state && state->status == RequestStatus::Active;
}

bool Scheduler::has_started_execution(const StatePtr& state) noexcept {
    return state && (state->phase != GenerationPhase::Prefill || state->prefill_cursor > 0);
}

bool Scheduler::sampling_compatible(const SamplingParams& lhs, const SamplingParams& rhs) noexcept {
    return lhs.temperature == rhs.temperature && lhs.top_p == rhs.top_p && lhs.top_k == rhs.top_k &&
           lhs.seed == rhs.seed;
}

void Scheduler::build_decode_batch(BatchBuildContext& ctx) {
    for (SequenceId id : active_order_) {
        if (ctx.compute_budget <= 0) break;
        auto it = active_.find(id);
        if (it == active_.end()) continue;

        auto& state = it->second;
        if (!is_schedulable_state(state) || !is_decode_phase(state->phase)) continue;
        if (state->tokens_generated >= state->sampling.max_tokens) {
            state->status = RequestStatus::Finished;
            continue;
        }

        int expected_ctx = 0;
        int32_t input_token = 0;
        bool write_kv = true;
        if (state->phase == GenerationPhase::Bootstrap) {
            // Bootstrap decode (write_kv=false) samples from an already-cached
            // prompt; it writes no KV and allocates no block.
            expected_ctx = static_cast<int>(state->prompt_tokens.size()) + state->tokens_generated -
                           state->generated_in_prompt;
            input_token = state->prompt_tokens.back();
            write_kv = false;
        } else {
            if (state->last_token < 0) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }
            expected_ctx = static_cast<int>(state->prompt_tokens.size()) + state->tokens_generated -
                           state->generated_in_prompt - 1;
            if (expected_ctx < 0) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }
            input_token = state->last_token;
        }

        if (expected_ctx >= state->max_context_len) {
            send_terminal_event(state);
            continue;
        }

        int decode_blocks = 0;
        if (write_kv) {
            decode_blocks = (expected_ctx % ctx.block_size == 0) ? 1 : 0;
            if (decode_blocks > ctx.budget) {
                budget_blocked_.push_back(id);
                continue;
            }
        }

        if (ctx.sampling_set) {
            if (!sampling_compatible(state->sampling, ctx.batch.sampling)) continue;
        } else {
            ctx.batch.sampling = state->sampling;
            ctx.sampling_set = true;
        }

        ctx.batch.items.push_back(
            DecodeOneToken{id, input_token, std::optional<int>(expected_ctx), write_kv});
        ctx.budget -= decode_blocks;
        ctx.compute_budget -= 1;
        ctx.included.push_back(id);
    }
}

void Scheduler::build_prefill_batch(BatchBuildContext& ctx) {
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const int configured_chunk_size = engine_config_.prefill_chunk_size;

    for (size_t idx = 0; idx < active_order_.size(); ++idx) {
        SequenceId id = active_order_[idx];
        auto it = active_.find(id);
        if (it == active_.end()) continue;

        auto& state = it->second;
        if (!is_schedulable_state(state) || !is_prefill_phase(state->phase)) continue;
        const bool already_started = has_started_execution(state);
        if (!already_started &&
            ctx.started_active >= engine_config_.max_active_scheduled_sequences) {
            continue;
        }

        int prompt_len = static_cast<int>(state->prompt_tokens.size());
        if (state->prefill_cursor < 0) state->prefill_cursor = 0;
        if (state->prefill_cursor > prompt_len) {
            send_error_event(state, ErrorCode::InvalidArgument);
            continue;
        }

        int remaining = prompt_len - state->prefill_cursor;
        if (remaining <= 0) {
            state->phase = state->generated_tokens.empty() ? GenerationPhase::Bootstrap
                                                           : GenerationPhase::Decode;
            continue;
        }

        // Chunked prefill: limit tokens per batch to bound per-step latency.
        // A configured size of 0 disables chunking for benchmark ablations.
        int chunk_len = remaining;
        if (configured_chunk_size > 0) {
            chunk_len = std::min(remaining, configured_chunk_size);
        }
        if (ctx.compute_budget <= 0) break;
        chunk_len = std::min(chunk_len, ctx.compute_budget);

        // Incremental blocks: only count blocks beyond what's already owned.
        int cur_blocks = ceil_div(state->prefill_cursor, ctx.block_size);
        int total_blocks = ceil_div(state->prefill_cursor + chunk_len, ctx.block_size);
        int additional = total_blocks - cur_blocks;
        if (additional > ctx.budget) {
            // Shrink chunk to fit in available budget.
            int max_tokens_fit = (cur_blocks + ctx.budget) * ctx.block_size - state->prefill_cursor;
            if (max_tokens_fit <= 0) {
                budget_blocked_.push_back(id);
                continue;
            }
            chunk_len = max_tokens_fit;
            total_blocks = ceil_div(state->prefill_cursor + chunk_len, ctx.block_size);
            additional = total_blocks - cur_blocks;
        }

        // Shared sampling per batch: use first item's sampling; skip
        // items with incompatible params.
        if (ctx.sampling_set) {
            if (!sampling_compatible(state->sampling, ctx.batch.sampling)) continue;
        } else {
            ctx.batch.sampling = state->sampling;
            ctx.sampling_set = true;
        }

        bool is_final = (state->prefill_cursor + chunk_len >= prompt_len);
        bool needs_sample = is_final;
        if (state->phase == GenerationPhase::ReplayPrefill && !state->generated_tokens.empty()) {
            needs_sample = false;
        }
        ctx.batch.items.push_back(PrefillChunk{id, TokenSpan{state->prefill_cursor, chunk_len},
                                               std::optional<int>(state->prefill_cursor),
                                               needs_sample});
        ctx.budget -= additional;
        ctx.compute_budget -= chunk_len;
        if (!already_started) ++ctx.started_active;
        ctx.included.push_back(id);
    }
}

void Scheduler::reorder_after_batch(const std::vector<SequenceId>& included) {
    std::unordered_set<SequenceId> inc(included.begin(), included.end());
    std::deque<SequenceId> keep;
    for (SequenceId id : active_order_) {
        if (inc.count(id)) continue;
        auto it = active_.find(id);
        if (it == active_.end()) continue;
        auto& state = it->second;
        if (!is_schedulable_state(state)) continue;
        keep.push_back(id);
    }
    for (SequenceId id : included) {
        if (active_.count(id)) keep.push_back(id);
    }
    active_order_ = std::move(keep);
}

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
        if (!is_schedulable_state(state)) continue;

        bool has_new_token = false;
        bool is_final_chunk = false;

        if (wr.kind == WorkKind::PrefillChunk) {
            int prompt_len = static_cast<int>(state->prompt_tokens.size());
            int remaining = prompt_len - state->prefill_cursor;

            int expected_chunk_len = 0;
            std::visit(
                [&](const auto& w) {
                    using T = std::decay_t<decltype(w)>;
                    if constexpr (std::is_same_v<T, PrefillChunk>)
                        expected_chunk_len = w.prompt_span.length;
                },
                original_item);

            if (remaining <= 0 || wr.tokens_consumed <= 0 || wr.tokens_consumed > remaining ||
                wr.tokens_consumed != expected_chunk_len) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }

            is_final_chunk = (state->prefill_cursor + wr.tokens_consumed >= prompt_len);

            const bool replay_prefill = state->phase == GenerationPhase::ReplayPrefill;
            state->prefill_cursor += wr.tokens_consumed;
            const bool prefill_complete = state->prefill_cursor >= prompt_len;
            if (prefill_complete) state->phase = GenerationPhase::Decode;

            // Only sample/send token on the final prefill chunk.
            // Intermediate chunks write KV cache but produce no output token.
            if (!is_final_chunk) {
                if (!wr.sampled_tokens.empty() || wr.eos) {
                    send_error_event(state, ErrorCode::BatchTranslationFailed);
                    continue;
                }
            } else {
                const bool replay_done_without_sample = replay_prefill &&
                                                        !state->generated_tokens.empty() &&
                                                        wr.sampled_tokens.empty() && !wr.eos;
                if (wr.sampled_tokens.empty() && !wr.eos && !replay_done_without_sample) {
                    send_error_event(state, ErrorCode::BatchTranslationFailed);
                    continue;
                }
                if (!wr.sampled_tokens.empty()) {
                    state->last_token = wr.sampled_tokens[0];
                    state->tokens_generated++;
                    state->generated_tokens.push_back(wr.sampled_tokens[0]);
                    has_new_token = true;
                }
                state->phase = GenerationPhase::Decode;
            }
        } else {
            // DecodeOneToken represents one decode step; bootstrap decode simply
            // uses write_kv=false.
            state->phase = GenerationPhase::Decode;
            if (state->prefill_cursor < static_cast<int>(state->prompt_tokens.size())) {
                state->prefill_cursor = static_cast<int>(state->prompt_tokens.size());
            }

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
                state->generated_tokens.push_back(wr.sampled_tokens[0]);
                has_new_token = true;
            }
        }

        // eos only valid for decode items and final prefill chunks.
        if (wr.eos && (wr.kind == WorkKind::DecodeOneToken || is_final_chunk))
            state->status = RequestStatus::Finished;
        if (state->sampling.max_tokens <= 0) state->status = RequestStatus::Finished;
        if (state->tokens_generated >= state->sampling.max_tokens)
            state->status = RequestStatus::Finished;

        if (has_new_token) {
            send_token_event(state);
        } else if (state->status == RequestStatus::Finished) {
            send_terminal_event(state);
        }
    }

    co_return;
}

void Scheduler::filter_active_order() {
    std::deque<SequenceId> new_order;
    for (SequenceId id : active_order_) {
        auto it = active_.find(id);
        if (it == active_.end()) continue;
        auto& state = it->second;
        if (!state) continue;
        if (!is_schedulable_state(state)) continue;
        new_order.push_back(id);
    }
    active_order_ = std::move(new_order);
}

asio::awaitable<void> Scheduler::cleanup_terminal_requests() {
    std::vector<SequenceId> to_release;
    for (const auto& [seq_id, state] : active_) {
        if (state && !is_schedulable_state(state)) {
            to_release.push_back(seq_id);
        }
    }

    for (SequenceId seq_id : to_release) {
        auto it = active_.find(seq_id);
        const bool cancelled =
            it != active_.end() && it->second &&
            (it->second->status == RequestStatus::Cancelled || it->second->sink_disconnected);
        auto r = cancelled ? co_await executor_.abort_sequence(seq_id)
                           : co_await executor_.release_sequence(seq_id);
        if (!r) {
            ccLog::warn("sequence cleanup failed seq={} err={}", seq_id,
                        static_cast<int>(r.error()));
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

asio::awaitable<void> Scheduler::suspend_sequence_for_replay(StatePtr& state) {
    if (!is_schedulable_state(state)) co_return;

    std::vector<int32_t> replay_prompt = state->initial_prompt_tokens;
    int generated_in_prompt = 0;
    if (!state->generated_tokens.empty()) {
        generated_in_prompt = static_cast<int>(state->generated_tokens.size()) - 1;
        replay_prompt.insert(replay_prompt.end(), state->generated_tokens.begin(),
                             state->generated_tokens.end() - 1);
    }

    auto r =
        co_await executor_.suspend_sequence(state->seq_id, replay_prompt, state->max_context_len);
    if (!r) {
        send_error_event(state, r.error());
        co_return;
    }

    state->prompt_tokens = std::move(replay_prompt);
    state->generated_in_prompt = generated_in_prompt;
    state->prefill_cursor = 0;
    state->phase = GenerationPhase::ReplayPrefill;

    const int prompt_len = static_cast<int>(state->prompt_tokens.size());
    if (r->prompt_processed > 0 && r->prompt_processed < prompt_len) {
        state->prefill_cursor = r->prompt_processed;
    } else if (r->prompt_processed >= prompt_len) {
        state->prefill_cursor = prompt_len;
        state->phase =
            state->generated_tokens.empty() ? GenerationPhase::Bootstrap : GenerationPhase::Decode;
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
        if (state && is_schedulable_state(state)) {
            send_error_event(state, shutdown_err);
        }

        const bool cancelled =
            state && (state->status == RequestStatus::Cancelled || state->sink_disconnected);
        auto r = cancelled ? co_await executor_.abort_sequence(seq_id)
                           : co_await executor_.release_sequence(seq_id);
        if (!r) {
            ccLog::warn("sequence cleanup failed seq={} err={}", seq_id,
                        static_cast<int>(r.error()));
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

}  // namespace ccinfer
