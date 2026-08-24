#include "scheduler/scheduler.h"

#include <algorithm>
#include <chrono>
#include <iterator>
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

#include "engine/engine_core.h"
#include "facade/log.h"

namespace ccinfer {

namespace {
using asio::as_tuple;
using asio::deferred;

int generated_token_count(const RequestState& state) noexcept {
    return static_cast<int>(state.generated_tokens.size());
}

}  // namespace

Scheduler::Scheduler(asio::io_context& io, Executor& executor, EngineConfig config)
    : io_(io),
      executor_(executor),
      engine_config_(config),
      idle_timer_(io),
      core_(std::make_unique<EngineCore>(io, *this, executor, config)) {}

Scheduler::~Scheduler() = default;

void Scheduler::submit(SchedulerRequest req) {
    if (!accepting_.load()) {
        send_event(req.sink, std::unexpected(ErrorCode::ServerShuttingDown));
        return;
    }

    asio::post(io_, [this, req = std::move(req)]() mutable {
        if (!accepting_.load()) {
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
    std::lock_guard lock(shutdown_mutex_);
    if (shutdown_promise_) return;

    bool expected = false;
    if (!accepting_.compare_exchange_strong(expected, true)) return;
    core_->start();
}

void Scheduler::shutdown() { shutdown_async(); }

std::shared_future<void> Scheduler::shutdown_async() {
    std::lock_guard lock(shutdown_mutex_);
    if (shutdown_promise_) return shutdown_future_;

    shutdown_promise_ = std::make_unique<std::promise<void>>();
    shutdown_future_ = shutdown_promise_->get_future().share();
    const bool was_running = accepting_.exchange(false);

    asio::post(io_, [this, was_running] {
        core_->request_shutdown();
        if (!was_running) complete_shutdown_on_scheduler_thread();
    });
    return shutdown_future_;
}

void Scheduler::submit_on_scheduler_thread(SchedulerRequest req) {
    if (requests_.count(req.request_id)) {
        send_event(req.sink, std::unexpected(ErrorCode::InvalidArgument));
        return;
    }
    if (waiting_.size() >= static_cast<std::size_t>(engine_config_.max_pending_requests)) {
        send_event(req.sink, std::unexpected(ErrorCode::RequestQueueFull));
        return;
    }
    if (req.max_context_len <= 0) req.max_context_len = engine_config_.default_max_context_len;

    auto request = std::make_shared<RequestState>();
    request->request_id = req.request_id;
    request->initial_prompt_tokens = req.prompt_tokens;
    request->prompt_tokens = std::move(req.prompt_tokens);
    request->sampling = req.sampling;
    request->max_context_len = req.max_context_len;
    request->sink = std::move(req.sink);
    requests_.emplace(request->request_id, request);
    waiting_.push_back(std::move(request));
    wake_on_scheduler_thread();
}

void Scheduler::cancel_on_scheduler_thread(const std::string& request_id) {
    auto it = requests_.find(request_id);
    if (it != requests_.end()) {
        auto& state = *it->second;
        if (state.status == RequestStatus::Active) {
            state.status = RequestStatus::Cancelled;
        }
        return;
    }
}

void Scheduler::wake_on_scheduler_thread() { idle_timer_.cancel(); }

void Scheduler::complete_shutdown_on_scheduler_thread() {
    if (shutdown_done_sent_) return;
    shutdown_done_sent_ = true;
    if (shutdown_promise_) shutdown_promise_->set_value();
}

bool Scheduler::send_event(const TokenSink& sink, Result<GeneratedToken> result) {
    if (sink.channel.expired()) return false;

    asio::post(sink.executor, [weak = sink.channel, result = std::move(result),
                               on_send_failed = sink.on_send_failed]() mutable {
        auto channel = weak.lock();
        if (!channel) {
            if (on_send_failed) on_send_failed();
            return;
        }
        channel->async_send(boost::system::error_code{}, std::move(result),
                            [on_send_failed](boost::system::error_code ec) {
                                if (ec && on_send_failed) on_send_failed();
                            });
    });
    return true;
}

bool Scheduler::send_token_event(RequestState& state) {
    GeneratedToken token;
    token.token_id = state.generated_tokens.back();
    token.has_token = true;
    token.finished = state.status == RequestStatus::Finished;
    const bool sent = send_event(state.sink, std::move(token));
    if (!sent) {
        state.sink_disconnected = true;
        if (state.status == RequestStatus::Active) state.status = RequestStatus::Cancelled;
    }
    return sent;
}

bool Scheduler::send_terminal_event(RequestState& state) {
    state.status = RequestStatus::Finished;
    const bool sent = send_event(state.sink, GeneratedToken{.has_token = false, .finished = true});
    if (!sent) state.sink_disconnected = true;
    return sent;
}

bool Scheduler::send_error_event(RequestState& state, ErrorCode err) {
    state.status = RequestStatus::Failed;
    const bool sent = send_event(state.sink, std::unexpected(err));
    if (!sent) state.sink_disconnected = true;
    return sent;
}

asio::awaitable<bool> Scheduler::admit_one_waiting(BatchBuildContext& ctx) {
    if (ctx.token_budget <= 0 || waiting_.empty() ||
        running_.size() >= static_cast<std::size_t>(engine_config_.max_running_requests)) {
        co_return false;
    }

    if (ctx.sampling_set && !sampling_compatible(waiting_.front()->sampling, ctx.batch.sampling)) {
        co_return false;
    }

    auto request = waiting_.front();
    waiting_.pop_front();
    auto& state = *request;
    if (state.status == RequestStatus::Cancelled) {
        send_event(state.sink, std::unexpected(ErrorCode::RequestCancelled));
        erase_request(request);
        co_return true;
    }

    if (state.sampling.max_tokens <= 0) {
        send_event(state.sink, GeneratedToken{.has_token = false, .finished = true});
        erase_request(request);
        co_return true;
    }

    SequenceInitialState initial_state;
    initial_state.tokens_generated = generated_token_count(state);
    initial_state.max_tokens = state.sampling.max_tokens;
    if (!state.generated_tokens.empty()) initial_state.last_token = state.generated_tokens.back();
    auto result = co_await executor_.admit_sequence(state.prompt_tokens, state.max_context_len,
                                                    initial_state);

    if (!accepting_.load()) {
        if (result) co_await executor_.release_sequence(result->seq_id);
        send_event(state.sink, std::unexpected(ErrorCode::ServerShuttingDown));
        erase_request(request);
        co_return true;
    }

    if (state.status == RequestStatus::Cancelled) {
        if (result) co_await executor_.abort_sequence(result->seq_id);
        send_event(state.sink, std::unexpected(ErrorCode::RequestCancelled));
        erase_request(request);
        co_return true;
    }

    if (!result) {
        if (result.error() == ErrorCode::KVBlockExhausted ||
            result.error() == ErrorCode::MaxSequencesReached) {
            waiting_.push_front(request);
            co_return false;
        }
        send_event(state.sink, std::unexpected(result.error()));
        erase_request(request);
        co_return true;
    }

    state.scheduling.emplace();
    auto& scheduling = *state.scheduling;
    scheduling.seq_id = result->seq_id;

    const int prompt_len = static_cast<int>(state.prompt_tokens.size());
    if (result->prompt_processed > 0 && result->prompt_processed < prompt_len) {
        scheduling.cursor.prefill_cursor = result->prompt_processed;
        scheduling.cursor.phase = state.generated_tokens.empty() ? GenerationPhase::Prefill
                                                                 : GenerationPhase::ReplayPrefill;
    } else if (result->prompt_processed >= prompt_len) {
        scheduling.cursor.prefill_cursor = prompt_len;
        scheduling.cursor.phase =
            state.generated_tokens.empty() ? GenerationPhase::Bootstrap : GenerationPhase::Decode;
    }
    by_seq_id_[scheduling.seq_id] = request;
    running_.push_back(std::move(request));

    build_state_work(ctx, state);
    co_return true;
}

asio::awaitable<ScheduledBatch> Scheduler::schedule_step() {
    BatchBuildContext ctx;
    ctx.batch.batch_id = next_batch_id_++;
    ctx.token_budget = engine_config_.max_token_budget;
    const auto cap = executor_.capacity();
    ctx.block_size = cap.block_size;

    if (ctx.block_size <= 0) {
        for (auto request_it : running_) {
            auto& state = *request_it;
            if (is_schedulable_state(state)) send_error_event(state, ErrorCode::InternalError);
        }
        co_return std::move(ctx.batch);
    }

    build_running_batch(ctx);

    while (accepting_.load() && ctx.token_budget > 0 &&
           running_.size() < static_cast<std::size_t>(engine_config_.max_running_requests) &&
           !waiting_.empty()) {
        if (!co_await admit_one_waiting(ctx)) break;
    }

    co_return std::move(ctx.batch);
}

void Scheduler::build_running_batch(BatchBuildContext& ctx) {
    for (auto request_it : running_) {
        if (ctx.token_budget <= 0) break;
        build_state_work(ctx, *request_it);
    }
}

bool Scheduler::build_state_work(BatchBuildContext& ctx, RequestState& state) {
    if (!state.scheduling) return false;
    auto& scheduling = *state.scheduling;

    auto select_sampling = [&](const RequestState& state) {
        if (!ctx.sampling_set) {
            ctx.batch.sampling = state.sampling;
            ctx.sampling_set = true;
            return true;
        }
        return sampling_compatible(state.sampling, ctx.batch.sampling);
    };

    if (ctx.token_budget <= 0 || !is_schedulable_state(state)) return false;

    if (generated_token_count(state) >= state.sampling.max_tokens) {
        send_terminal_event(state);
        return false;
    }

    if (is_prefill_phase(scheduling.cursor.phase)) {
        const int prompt_len = static_cast<int>(state.prompt_tokens.size());
        const int prompt_start =
            scheduling.cursor.prefill_cursor + scheduling.reservation.reserved_prompt_tokens;
        if (prompt_start >= prompt_len) return false;

        int chunk_len = prompt_len - prompt_start;
        if (engine_config_.max_seq_prefill_tokens > 0) {
            chunk_len = std::min(chunk_len, engine_config_.max_seq_prefill_tokens);
        }
        chunk_len = std::min(chunk_len, ctx.token_budget);
        if (chunk_len <= 0 || !select_sampling(state)) return false;

        const bool final_chunk = prompt_start + chunk_len >= prompt_len;
        const bool replay = scheduling.cursor.phase == GenerationPhase::ReplayPrefill;
        const bool needs_sample = final_chunk && !(replay && !state.generated_tokens.empty());
        ctx.batch.items.push_back(PrefillChunk{
            scheduling.seq_id, TokenSpan{prompt_start, chunk_len}, std::nullopt, needs_sample});
        ctx.token_budget -= chunk_len;
        scheduling.reservation.reserved_prompt_tokens += chunk_len;
        if (needs_sample) ++scheduling.reservation.reserved_generation_tokens;
        ++scheduling.reservation.execution_leases;
        return true;
    }

    if (!is_decode_phase(scheduling.cursor.phase) || !select_sampling(state)) return false;
    if (generated_token_count(state) + scheduling.reservation.reserved_generation_tokens >=
        state.sampling.max_tokens)
        return false;

    int32_t input_token = 0;
    bool write_kv = true;
    bool late_bind = false;
    if (scheduling.cursor.phase == GenerationPhase::Bootstrap) {
        if (scheduling.reservation.reserved_generation_tokens > 0) return false;
        if (state.prompt_tokens.empty()) {
            send_error_event(state, ErrorCode::BatchTranslationFailed);
            return false;
        }
        input_token = state.prompt_tokens.back();
        write_kv = false;
    } else {
        // The token is deliberately not selected here. Worker resolves it
        // from the persistent state immediately before execution.
        late_bind = true;
    }

    const int committed_context = static_cast<int>(state.prompt_tokens.size()) +
                                  generated_token_count(state) -
                                  scheduling.cursor.generated_tokens_in_prompt;
    if (committed_context + scheduling.reservation.reserved_generation_tokens >=
        state.max_context_len) {
        if (scheduling.reservation.reserved_generation_tokens == 0) send_terminal_event(state);
        return false;
    }

    ctx.batch.items.push_back(
        DecodeOneToken{scheduling.seq_id, input_token, std::nullopt, late_bind, write_kv});
    --ctx.token_budget;
    ++scheduling.reservation.reserved_generation_tokens;
    ++scheduling.reservation.execution_leases;
    return true;
}

bool Scheduler::has_schedulable_work() const {
    for (auto request_it : running_) {
        const auto& state = *request_it;
        if (!state.scheduling) continue;
        const auto& scheduling = *state.scheduling;
        if (!is_schedulable_state(state)) continue;
        if (is_prefill_phase(scheduling.cursor.phase)) {
            if (scheduling.cursor.prefill_cursor + scheduling.reservation.reserved_prompt_tokens <
                static_cast<int>(state.prompt_tokens.size()))
                return true;
        } else if (generated_token_count(state) +
                       scheduling.reservation.reserved_generation_tokens <
                   state.sampling.max_tokens) {
            if (scheduling.cursor.phase != GenerationPhase::Bootstrap ||
                scheduling.reservation.reserved_generation_tokens == 0)
                return true;
        }
    }
    return false;
}

bool Scheduler::is_schedulable_state(const RequestState& state) noexcept {
    return state.status == RequestStatus::Active && state.scheduling.has_value();
}

bool Scheduler::is_prefill_phase(GenerationPhase phase) noexcept {
    return phase == GenerationPhase::Prefill || phase == GenerationPhase::ReplayPrefill;
}

bool Scheduler::is_decode_phase(GenerationPhase phase) noexcept {
    return phase == GenerationPhase::Bootstrap || phase == GenerationPhase::Decode;
}

bool Scheduler::sampling_compatible(const SamplingParams& lhs, const SamplingParams& rhs) noexcept {
    return lhs.temperature == rhs.temperature && lhs.top_p == rhs.top_p && lhs.top_k == rhs.top_k &&
           lhs.seed == rhs.seed;
}

void Scheduler::retire_reservation(RequestState& state, const WorkItem& item) {
    if (!state.scheduling) return;
    auto& reservation = state.scheduling->reservation;
    if (reservation.execution_leases > 0) --reservation.execution_leases;

    std::visit(
        [&](const auto& work) {
            using T = std::decay_t<decltype(work)>;
            if constexpr (std::is_same_v<T, PrefillChunk>) {
                reservation.reserved_prompt_tokens =
                    std::max(0, reservation.reserved_prompt_tokens - work.prompt_span.length);
                if (work.needs_sample && reservation.reserved_generation_tokens > 0)
                    --reservation.reserved_generation_tokens;
            } else {
                if (reservation.reserved_generation_tokens > 0)
                    --reservation.reserved_generation_tokens;
            }
        },
        item);
}

asio::awaitable<void> Scheduler::update_from_output(const ScheduledBatch& batch,
                                                    const BatchResult& result) {
    if (result.batch_id != batch.batch_id || result.items.size() != batch.items.size()) {
        co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
        co_return;
    }

    std::unordered_set<int> seen_item_indices;
    for (const auto& work_result : result.items) {
        if (work_result.item_index < 0 ||
            static_cast<std::size_t>(work_result.item_index) >= batch.items.size()) {
            co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
            co_return;
        }
        if (!seen_item_indices.insert(work_result.item_index).second) {
            co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
            co_return;
        }

        const auto& original = batch.items[work_result.item_index];
        if (work_sequence_id(original) != work_result.seq_id ||
            work_kind(original) != work_result.kind) {
            co_await fail_batch(batch, ErrorCode::BatchTranslationFailed);
            co_return;
        }
    }

    for (const auto& work_result : result.items) {
        auto it = by_seq_id_.find(work_result.seq_id);
        if (it == by_seq_id_.end()) continue;
        auto& state = *it->second;
        if (!state.scheduling) continue;
        auto& scheduling = *state.scheduling;
        const auto& original = batch.items[work_result.item_index];
        retire_reservation(state, original);

        // A later batch may have been dispatched before an earlier batch
        // produced EOS or otherwise changed the request state. It still owns
        // its reservation until retirement, but its result is intentionally
        // ignored.
        if (work_result.stale || !is_schedulable_state(state)) continue;

        bool final_prefill = false;
        if (work_result.kind == WorkKind::PrefillChunk) {
            const auto& chunk = std::get<PrefillChunk>(original);
            const int prompt_len = static_cast<int>(state.prompt_tokens.size());
            const bool replay_prefill = scheduling.cursor.phase == GenerationPhase::ReplayPrefill;
            if (work_result.tokens_consumed != chunk.prompt_span.length ||
                work_result.tokens_consumed <= 0 ||
                scheduling.cursor.prefill_cursor != chunk.prompt_span.start ||
                scheduling.cursor.prefill_cursor + work_result.tokens_consumed > prompt_len) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }
            final_prefill =
                scheduling.cursor.prefill_cursor + work_result.tokens_consumed >= prompt_len;
            scheduling.cursor.prefill_cursor += work_result.tokens_consumed;
            if (final_prefill) scheduling.cursor.phase = GenerationPhase::Decode;

            if (!final_prefill) {
                if (!work_result.sampled_tokens.empty() || work_result.eos)
                    send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }

            const bool replay_without_sample = replay_prefill && !state.generated_tokens.empty() &&
                                               work_result.sampled_tokens.empty() &&
                                               !work_result.eos;
            if (work_result.sampled_tokens.empty() && !work_result.eos && !replay_without_sample) {
                send_error_event(state, ErrorCode::BatchTranslationFailed);
                continue;
            }
        } else if (work_result.tokens_consumed != 1) {
            send_error_event(state, ErrorCode::BatchTranslationFailed);
            continue;
        }

        const bool finished =
            work_result.eos ||
            generated_token_count(state) + static_cast<int>(!work_result.sampled_tokens.empty()) >=
                state.sampling.max_tokens;
        if (!work_result.sampled_tokens.empty()) {
            state.generated_tokens.push_back(work_result.sampled_tokens.front());
        }
        if (finished || generated_token_count(state) >= state.sampling.max_tokens)
            state.status = RequestStatus::Finished;
        if (!work_result.sampled_tokens.empty()) send_token_event(state);
        if (state.status == RequestStatus::Finished && work_result.sampled_tokens.empty()) {
            send_terminal_event(state);
        }
    }

    for (auto request_it : running_) {
        auto& state = *request_it;
        if (!state.scheduling) continue;
        auto& scheduling = *state.scheduling;
        if (!scheduling.cursor.replay_pending || scheduling.reservation.execution_leases != 0 ||
            !is_schedulable_state(state))
            continue;
        scheduling.cursor.replay_pending = false;
        co_await suspend_sequence_for_replay(state);
    }
}

void Scheduler::mark_dispatch_failed(const ScheduledBatch& batch, ErrorCode err) {
    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        auto it = by_seq_id_.find(seq_id);
        if (it == by_seq_id_.end()) continue;
        retire_reservation(*it->second, item);
        if (err == ErrorCode::KVBlockExhausted) continue;
        send_error_event(*it->second, err);
    }
}

void Scheduler::erase_request(const RequestPtr& request) {
    if (request->scheduling) by_seq_id_.erase(request->scheduling->seq_id);
    requests_.erase(request->request_id);
}

asio::awaitable<void> Scheduler::cleanup_terminal_requests() {
    std::vector<RunningIterator> to_release;
    for (auto running_it = running_.begin(); running_it != running_.end(); ++running_it) {
        const auto& state = **running_it;
        if (state.scheduling && state.status != RequestStatus::Active &&
            state.scheduling->reservation.execution_leases == 0)
            to_release.push_back(running_it);
    }

    for (auto running_it : to_release) {
        auto request_it = *running_it;
        auto& state = *request_it;
        const SequenceId seq_id = state.scheduling->seq_id;
        const bool cancelled = state.status == RequestStatus::Cancelled || state.sink_disconnected;
        auto result = cancelled ? co_await executor_.abort_sequence(seq_id)
                                : co_await executor_.release_sequence(seq_id);
        if (!result) {
            ccLog::warn("sequence cleanup failed seq={} err={}", seq_id,
                        static_cast<int>(result.error()));
        }
        by_seq_id_.erase(seq_id);
        running_.erase(running_it);
        erase_request(request_it);
    }
}

asio::awaitable<void> Scheduler::fail_batch(const ScheduledBatch& batch, ErrorCode err) {
    std::unordered_set<SequenceId> seen;
    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        if (!seen.insert(seq_id).second) continue;
        auto it = by_seq_id_.find(seq_id);
        if (it == by_seq_id_.end()) continue;
        retire_reservation(*it->second, item);
        if (it->second->status == RequestStatus::Active) send_error_event(*it->second, err);
    }
    co_return;
}

asio::awaitable<void> Scheduler::handle_batch_error(const ScheduledBatch& batch, ErrorCode err) {
    if (err != ErrorCode::KVBlockExhausted) {
        co_await fail_batch(batch, err);
        co_return;
    }

    std::unordered_set<SequenceId> seen;
    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        if (!seen.insert(seq_id).second) continue;
        auto it = by_seq_id_.find(seq_id);
        if (it == by_seq_id_.end()) continue;
        auto& state = *it->second;
        if (!state.scheduling) continue;
        retire_reservation(state, item);
        state.scheduling->cursor.replay_pending = true;
        if (state.scheduling->reservation.execution_leases == 0) {
            state.scheduling->cursor.replay_pending = false;
            co_await suspend_sequence_for_replay(state);
        }
    }
}

asio::awaitable<void> Scheduler::suspend_sequence_for_replay(RequestState& state) {
    if (!is_schedulable_state(state) || state.scheduling->reservation.execution_leases != 0)
        co_return;

    auto& scheduling = *state.scheduling;

    std::vector<int32_t> replay_prompt = state.initial_prompt_tokens;
    int generated_tokens_in_prompt = 0;
    if (!state.generated_tokens.empty()) {
        generated_tokens_in_prompt = static_cast<int>(state.generated_tokens.size()) - 1;
        replay_prompt.insert(replay_prompt.end(), state.generated_tokens.begin(),
                             state.generated_tokens.end() - 1);
    }

    auto result = co_await executor_.suspend_sequence(scheduling.seq_id, replay_prompt,
                                                      state.max_context_len);
    if (!result) {
        send_error_event(state, result.error());
        co_return;
    }

    state.prompt_tokens = std::move(replay_prompt);
    scheduling.cursor.generated_tokens_in_prompt = generated_tokens_in_prompt;
    scheduling.reservation.reserved_prompt_tokens = 0;
    scheduling.reservation.reserved_generation_tokens = 0;
    scheduling.cursor.prefill_cursor = result->prompt_processed;
    scheduling.cursor.phase =
        result->prompt_processed >= static_cast<int>(state.prompt_tokens.size())
            ? (state.generated_tokens.empty() ? GenerationPhase::Bootstrap
                                              : GenerationPhase::Decode)
            : GenerationPhase::ReplayPrefill;
}

asio::awaitable<void> Scheduler::preempt_one_for_admission() {
    for (auto reverse_it = running_.rbegin(); reverse_it != running_.rend(); ++reverse_it) {
        auto running_it = std::prev(reverse_it.base());
        auto request_it = *running_it;
        auto& state = *request_it;
        if (!is_schedulable_state(state) || state.scheduling->reservation.execution_leases != 0)
            continue;

        auto result = co_await executor_.abort_sequence(state.scheduling->seq_id);
        if (!result) co_return;

        by_seq_id_.erase(state.scheduling->seq_id);
        state.scheduling.reset();
        running_.erase(running_it);
        waiting_.push_front(request_it);
        co_return;
    }
}

void Scheduler::fail_all_waiting(ErrorCode err) {
    while (!waiting_.empty()) {
        auto request_it = waiting_.front();
        waiting_.pop_front();
        send_event(request_it->sink, std::unexpected(err));
        erase_request(request_it);
    }
}

asio::awaitable<void> Scheduler::cleanup_all_running(ErrorCode shutdown_err) {
    std::vector<RunningIterator> states;
    states.reserve(running_.size());
    for (auto it = running_.begin(); it != running_.end(); ++it) states.push_back(it);

    for (auto running_it : states) {
        auto request_it = *running_it;
        auto& state = *request_it;
        if (!state.scheduling) continue;
        const SequenceId seq_id = state.scheduling->seq_id;
        if (state.status == RequestStatus::Active) send_error_event(state, shutdown_err);

        auto result = (state.status == RequestStatus::Cancelled || state.sink_disconnected)
                          ? co_await executor_.abort_sequence(seq_id)
                          : co_await executor_.release_sequence(seq_id);
        if (!result) {
            ccLog::warn("shutdown cleanup failed seq={} err={}", seq_id,
                        static_cast<int>(result.error()));
        }
        running_.erase(running_it);
        erase_request(request_it);
    }
}

asio::awaitable<void> Scheduler::wait_for_work() {
    if (!accepting_.load() || !waiting_.empty() || has_schedulable_work()) co_return;
    idle_timer_.expires_after(std::chrono::hours(24));
    auto [ec] = co_await idle_timer_.async_wait(as_tuple(deferred));
    (void)ec;
}

}  // namespace ccinfer
