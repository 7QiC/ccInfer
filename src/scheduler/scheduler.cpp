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

int kv_token_count(const WorkItem& item) noexcept {
    return std::visit(
        [](const auto& work) {
            if constexpr (std::is_same_v<std::decay_t<decltype(work)>, PrefillChunk>) {
                return work.prompt_span.length;
            } else {
                return work.write_kv ? 1 : 0;
            }
        },
        item);
}

}  // namespace

Scheduler::Scheduler(asio::io_context& io, Executor& executor, EngineConfig config)
    : io_(io),
      executor_(executor),
      engine_config_(config),
      idle_timer_(io),
      core_(std::make_unique<EngineCore>(io, *this, executor, config)),
      block_pool_(config.max_blocks, config.block_size),
      shutdown_future_(shutdown_promise_.get_future().share()) {}

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

Capacity Scheduler::capacity() const {
    const auto stats = block_pool_.stats();
    return Capacity{engine_config_.max_sequences,
                    static_cast<int>(running_.size() + skip_.size()),
                    stats.block_free,
                    stats.block_total,
                    stats.block_size,
                    stats.block_active,
                    stats.block_cached_idle,
                    stats.prefix.lookup_hits,
                    stats.prefix.lookup_misses,
                    stats.prefix.evictions,
                    static_cast<uint64_t>(stats.prefix.cached_blocks)};
}

void Scheduler::start() {
    if (shutdown_requested_.load()) return;

    bool expected = false;
    if (!accepting_.compare_exchange_strong(expected, true)) return;
    core_->start();
}

void Scheduler::shutdown() { shutdown_async(); }

std::shared_future<void> Scheduler::shutdown_async() {
    if (shutdown_requested_.exchange(true)) return shutdown_future_;
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
    shutdown_promise_.set_value();
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
    if (!sent && state.status == RequestStatus::Active) state.status = RequestStatus::Cancelled;
    return sent;
}

bool Scheduler::send_terminal_event(RequestState& state) {
    state.status = RequestStatus::Finished;
    return send_event(state.sink, GeneratedToken{.has_token = false, .finished = true});
}

bool Scheduler::send_error_event(RequestState& state, ErrorCode err) {
    state.status = RequestStatus::Failed;
    return send_event(state.sink, std::unexpected(err));
}

asio::awaitable<bool> Scheduler::admit_one_skipped(BatchBuildContext& ctx) {
    if (ctx.token_budget <= 0 || skip_.empty() ||
        running_.size() >= static_cast<std::size_t>(engine_config_.max_running_requests)) {
        co_return false;
    }

    auto request = skip_.front();
    skip_.pop_front();
    auto& state = *request;
    if (state.status == RequestStatus::Cancelled) {
        if (state.scheduling) {
            release_scheduling_blocks(state);
        }
        send_event(state.sink, std::unexpected(ErrorCode::RequestCancelled));
        erase_request(request);
        co_return true;
    }

    if (state.sampling.max_tokens <= 0) {
        if (state.scheduling) {
            release_scheduling_blocks(state);
        }
        send_event(state.sink, GeneratedToken{.has_token = false, .finished = true});
        erase_request(request);
        co_return true;
    }

    if (!state.scheduling) {
        waiting_.push_front(request);
        co_return false;
    }

    if (!accepting_.load()) {
        release_scheduling_blocks(state);
        send_event(state.sink, std::unexpected(ErrorCode::ServerShuttingDown));
        erase_request(request);
        co_return true;
    }

    if (generated_token_count(state) >= state.sampling.max_tokens) {
        release_scheduling_blocks(state);
        send_terminal_event(state);
        erase_request(request);
        co_return true;
    }

    by_seq_id_[state.scheduling->seq_id] = request;
    running_.push_back(std::move(request));
    build_state_work(ctx, state);
    co_return true;
}

asio::awaitable<bool> Scheduler::evict_one_skipped() {
    if (skip_.empty()) co_return false;
    auto request = skip_.front();
    skip_.pop_front();

    if (!request->scheduling) {
        waiting_.push_front(request);
        co_return true;
    }

    prepare_for_wait(*request);
    release_scheduling_blocks(*request);
    waiting_.push_front(request);
    co_return true;
}

asio::awaitable<bool> Scheduler::admit_one_waiting(BatchBuildContext& ctx) {
    if (ctx.token_budget <= 0 || waiting_.empty() ||
        running_.size() >= static_cast<std::size_t>(engine_config_.max_running_requests) ||
        running_.size() + skip_.size() >= static_cast<std::size_t>(engine_config_.max_sequences)) {
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

    if (state.scheduling) {
        // Defensive: a scheduled request should not be in waiting under the
        // skip/wait model. Release it and re-admit as a fresh sequence.
        release_scheduling_blocks(state);
    }

    const int prompt_len = static_cast<int>(state.prompt_tokens.size());
    if (prompt_len <= 0 || prompt_len > state.max_context_len ||
        prompt_len > block_pool_.max_blocks() * block_pool_.block_size()) {
        send_event(state.sink, std::unexpected(ErrorCode::InvalidArgument));
        erase_request(request);
        co_return true;
    }

    auto lookup = block_pool_.lookup_prefix_cache(state.prompt_tokens);
    const int blocks_needed =
        (prompt_len + block_pool_.block_size() - 1) / block_pool_.block_size();
    const int missing_blocks = blocks_needed - lookup.block_table.size();
    Result<BlockTable> allocated = BlockTable{};
    if (missing_blocks > 0) allocated = block_pool_.allocate_blocks(missing_blocks);
    if (!allocated) {
        block_pool_.release_blocks(lookup.block_table);
        if (allocated.error() == ErrorCode::KVBlockExhausted && co_await evict_one_skipped()) {
            waiting_.push_front(request);
            co_return false;
        }
        if (allocated.error() == ErrorCode::KVBlockExhausted) {
            waiting_.push_front(request);
            co_return false;
        }
        send_event(state.sink, std::unexpected(allocated.error()));
        erase_request(request);
        co_return true;
    }
    for (int i = 0; i < allocated->size(); ++i) lookup.block_table.push_back((*allocated)[i]);

    if (!accepting_.load()) {
        block_pool_.release_blocks(lookup.block_table);
        send_event(state.sink, std::unexpected(ErrorCode::ServerShuttingDown));
        erase_request(request);
        co_return true;
    }

    state.scheduling.emplace();
    auto& scheduling = *state.scheduling;
    scheduling.seq_id = next_seq_id_++;
    scheduling.block_table = std::move(lookup.block_table);
    scheduling.executed_frontier = lookup.prefix_hit_blocks * block_pool_.block_size();
    scheduling.parent_hash = lookup.parent_hash;
    scheduling.pending_hash_tokens = std::move(lookup.pending_tokens);
    if (!state.generated_tokens.empty()) {
        scheduling.cursor.generated_tokens_in_prompt = generated_token_count(state) - 1;
    }

    if (scheduling.executed_frontier > 0 && scheduling.executed_frontier < prompt_len) {
        scheduling.cursor.prefill_cursor = scheduling.executed_frontier;
        scheduling.cursor.phase = state.generated_tokens.empty() ? GenerationPhase::Prefill
                                                                 : GenerationPhase::ReplayPrefill;
    } else if (scheduling.executed_frontier >= prompt_len) {
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
    ctx.block_size = block_pool_.block_size();

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
           (!skip_.empty() || !waiting_.empty())) {
        if (!skip_.empty()) {
            if (co_await admit_one_skipped(ctx)) continue;
        }
        if (waiting_.empty() || !co_await admit_one_waiting(ctx)) break;
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
        }
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
        if (chunk_len <= 0) return false;
        select_sampling(state);

        const bool final_chunk = prompt_start + chunk_len >= prompt_len;
        const bool replay = scheduling.cursor.phase == GenerationPhase::ReplayPrefill;
        const bool needs_sample = final_chunk && !(replay && !state.generated_tokens.empty());
        std::vector<int32_t> chunk_tokens(state.prompt_tokens.begin() + prompt_start,
                                          state.prompt_tokens.begin() + prompt_start + chunk_len);
        ctx.batch.items.push_back(
            PrefillChunk{scheduling.seq_id, TokenSpan{prompt_start, chunk_len}, std::nullopt,
                         needs_sample, std::move(chunk_tokens), scheduling.block_table});
        auto& item = std::get<PrefillChunk>(ctx.batch.items.back());
        item.expected_context_len =
            scheduling.executed_frontier + scheduling.reservation.reserved_kv_tokens;
        ctx.token_budget -= chunk_len;
        scheduling.reservation.reserved_prompt_tokens += chunk_len;
        scheduling.reservation.reserved_kv_tokens += chunk_len;
        if (needs_sample) ++scheduling.reservation.reserved_generation_tokens;
        ++scheduling.reservation.execution_leases;
        return true;
    }

    if (!is_decode_phase(scheduling.cursor.phase)) return false;
    select_sampling(state);
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
        // The token may be produced by an earlier in-flight batch. Worker
        // resolves it from the latest token produced for this sequence.
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

    const int context_len =
        scheduling.executed_frontier + scheduling.reservation.reserved_kv_tokens;
    const int required_tokens = context_len + (write_kv ? 1 : 0);
    const int required_blocks =
        (required_tokens + block_pool_.block_size() - 1) / block_pool_.block_size();
    if (required_blocks > scheduling.block_table.size()) {
        auto blocks = block_pool_.allocate_blocks(required_blocks - scheduling.block_table.size());
        if (!blocks) return false;
        for (int i = 0; i < blocks->size(); ++i) scheduling.block_table.push_back((*blocks)[i]);
    }

    ctx.batch.items.push_back(DecodeOneToken{scheduling.seq_id, input_token, context_len, write_kv,
                                             late_bind, scheduling.block_table});
    --ctx.token_budget;
    ++scheduling.reservation.reserved_generation_tokens;
    if (write_kv) ++scheduling.reservation.reserved_kv_tokens;
    ++scheduling.reservation.execution_leases;
    return true;
}

bool Scheduler::has_schedulable_work() const {
    if (!skip_.empty() &&
        running_.size() < static_cast<std::size_t>(engine_config_.max_running_requests)) {
        return true;
    }
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

void Scheduler::prepare_for_wait(RequestState& state) {
    if (state.generated_tokens.empty()) return;
    state.prompt_tokens = state.initial_prompt_tokens;
    state.prompt_tokens.insert(state.prompt_tokens.end(), state.generated_tokens.begin(),
                               state.generated_tokens.end() - 1);
}

void Scheduler::release_scheduling_blocks(RequestState& state) {
    if (!state.scheduling) return;
    block_pool_.release_blocks(state.scheduling->block_table);
    state.scheduling.reset();
}

asio::awaitable<void> Scheduler::release_and_move_to_wait(const RequestPtr& request) {
    auto& state = *request;
    if (!state.scheduling) co_return;

    prepare_for_wait(state);
    const SequenceId seq_id = state.scheduling->seq_id;
    release_scheduling_blocks(state);
    by_seq_id_.erase(seq_id);
    for (auto rit = running_.begin(); rit != running_.end(); ++rit) {
        if (rit->get() == request.get()) {
            running_.erase(rit);
            break;
        }
    }
    waiting_.push_front(request);
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

    reservation.reserved_kv_tokens =
        std::max(0, reservation.reserved_kv_tokens - kv_token_count(item));
}

asio::awaitable<void> Scheduler::update_from_output(const ScheduledBatch& batch,
                                                    const BatchResult& result) {
    assert(result.batch_id == batch.batch_id);
    assert(result.items.size() == batch.items.size());

    for (const auto& work_result : result.items) {
        assert(work_result.item_index >= 0 &&
               static_cast<std::size_t>(work_result.item_index) < batch.items.size());
        const auto& original = batch.items[work_result.item_index];
        assert(work_sequence_id(original) == work_result.seq_id);
        assert(work_kind(original) == work_result.kind);
    }

    for (const auto& work_result : result.items)
        retire_work(batch.items[work_result.item_index], work_result);
    co_return;
}

void Scheduler::retire_work(const WorkItem& original, const WorkItemResult& work_result) {
    auto it = by_seq_id_.find(work_result.seq_id);
    if (it == by_seq_id_.end() || !it->second->scheduling) return;
    auto& state = *it->second;
    auto& scheduling = *state.scheduling;
    retire_reservation(state, original);

    // A later batch may have been dispatched before an earlier batch produced
    // EOS or otherwise changed the request state. Its reservation is retired,
    // but its result is intentionally ignored.
    if (work_result.stale || !is_schedulable_state(state)) return;

    bool final_prefill = false;
    const int context_before = std::visit(
        [](const auto& work) { return work.expected_context_len.value_or(0); }, original);
    const int kv_tokens = kv_token_count(original);
    if (work_result.kind == WorkKind::PrefillChunk) {
        const auto& chunk = std::get<PrefillChunk>(original);
        const int prompt_len = static_cast<int>(state.prompt_tokens.size());
        const bool replay_prefill = scheduling.cursor.phase == GenerationPhase::ReplayPrefill;
        if (work_result.tokens_consumed != chunk.prompt_span.length ||
            work_result.tokens_consumed <= 0 ||
            scheduling.cursor.prefill_cursor != chunk.prompt_span.start ||
            scheduling.cursor.prefill_cursor + work_result.tokens_consumed > prompt_len) {
            send_error_event(state, ErrorCode::BatchTranslationFailed);
            return;
        }
        final_prefill =
            scheduling.cursor.prefill_cursor + work_result.tokens_consumed >= prompt_len;
        scheduling.cursor.prefill_cursor += work_result.tokens_consumed;
        scheduling.executed_frontier =
            std::max(scheduling.executed_frontier, context_before + work_result.tokens_consumed);
        if (final_prefill) scheduling.cursor.phase = GenerationPhase::Decode;

        if (!final_prefill) {
            if (!work_result.sampled_tokens.empty() || work_result.eos)
                send_error_event(state, ErrorCode::BatchTranslationFailed);
            return;
        }

        const bool replay_without_sample = replay_prefill && !state.generated_tokens.empty() &&
                                           work_result.sampled_tokens.empty() && !work_result.eos;
        if (work_result.sampled_tokens.empty() && !work_result.eos && !replay_without_sample) {
            send_error_event(state, ErrorCode::BatchTranslationFailed);
            return;
        }
    } else if (work_result.tokens_consumed != 1) {
        send_error_event(state, ErrorCode::BatchTranslationFailed);
        return;
    } else {
        scheduling.executed_frontier =
            std::max(scheduling.executed_frontier, context_before + kv_tokens);
    }

    if (kv_tokens > 0) {
        const int pending_before = static_cast<int>(scheduling.pending_hash_tokens.size());
        int next_block_index = (context_before - pending_before) / block_pool_.block_size();
        std::vector<int32_t> written_tokens;
        if (const auto* chunk = std::get_if<PrefillChunk>(&original)) {
            written_tokens = chunk->tokens;
        } else if (!state.generated_tokens.empty()) {
            written_tokens.push_back(state.generated_tokens.back());
        }
        scheduling.pending_hash_tokens.insert(scheduling.pending_hash_tokens.end(),
                                              written_tokens.begin(), written_tokens.end());
        while (scheduling.pending_hash_tokens.size() >=
               static_cast<std::size_t>(block_pool_.block_size())) {
            std::vector<int32_t> full_block(
                scheduling.pending_hash_tokens.begin(),
                scheduling.pending_hash_tokens.begin() + block_pool_.block_size());
            const int block_index = next_block_index++;
            assert(block_index >= 0 && block_index < scheduling.block_table.size());
            scheduling.parent_hash = block_pool_.publish_full_block(
                scheduling.parent_hash, full_block, scheduling.block_table[block_index]);
            scheduling.pending_hash_tokens.erase(
                scheduling.pending_hash_tokens.begin(),
                scheduling.pending_hash_tokens.begin() + block_pool_.block_size());
        }
    }

    const bool finished =
        work_result.eos ||
        generated_token_count(state) + static_cast<int>(!work_result.sampled_tokens.empty()) >=
            state.sampling.max_tokens;
    if (!work_result.sampled_tokens.empty()) {
        state.generated_tokens.push_back(work_result.sampled_tokens.front());
        if (scheduling.cursor.phase == GenerationPhase::Bootstrap)
            scheduling.cursor.phase = GenerationPhase::Decode;
    }
    if (finished || generated_token_count(state) >= state.sampling.max_tokens)
        state.status = RequestStatus::Finished;
    if (!work_result.sampled_tokens.empty()) send_token_event(state);
    if (state.status == RequestStatus::Finished && work_result.sampled_tokens.empty()) {
        send_terminal_event(state);
    }
}

void Scheduler::mark_dispatch_failed(const ScheduledBatch& batch, ErrorCode err) {
    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        auto it = by_seq_id_.find(seq_id);
        if (it == by_seq_id_.end()) continue;
        retire_reservation(*it->second, item);
        if (err != ErrorCode::KVBlockExhausted) send_error_event(*it->second, err);
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
        release_scheduling_blocks(state);
        by_seq_id_.erase(seq_id);
        running_.erase(running_it);
        erase_request(request_it);
    }

    co_await cleanup_terminal_queue(skip_);
    co_await cleanup_terminal_queue(waiting_);
}

asio::awaitable<void> Scheduler::cleanup_terminal_queue(std::deque<RequestPtr>& queue) {
    std::vector<std::deque<RequestPtr>::iterator> to_release;
    for (auto it = queue.begin(); it != queue.end(); ++it) {
        if ((*it)->status != RequestStatus::Active) to_release.push_back(it);
    }
    for (auto it : to_release) {
        auto request = *it;
        if (request->scheduling) {
            const SequenceId seq_id = request->scheduling->seq_id;
            release_scheduling_blocks(*request);
            by_seq_id_.erase(seq_id);
        }
        queue.erase(it);
        erase_request(request);
    }
    co_return;
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

    // Free lower-priority skipped sequences before evicting running ones.
    while (co_await evict_one_skipped()) {
    }

    std::unordered_set<SequenceId> seen;
    std::vector<RequestPtr> to_wait;
    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        if (!seen.insert(seq_id).second) continue;
        auto it = by_seq_id_.find(seq_id);
        if (it == by_seq_id_.end()) continue;
        auto request = it->second;
        auto& state = *request;
        if (!state.scheduling) continue;
        retire_reservation(state, item);
        if (state.scheduling->reservation.execution_leases == 0 &&
            state.status == RequestStatus::Active) {
            to_wait.push_back(std::move(request));
        }
    }

    for (auto& request : to_wait) co_await release_and_move_to_wait(request);
}

asio::awaitable<void> Scheduler::preempt_one_for_admission() {
    for (auto reverse_it = running_.rbegin(); reverse_it != running_.rend(); ++reverse_it) {
        auto running_it = std::prev(reverse_it.base());
        auto request_it = *running_it;
        auto& state = *request_it;
        if (!is_schedulable_state(state) || state.scheduling->reservation.execution_leases != 0)
            continue;

        by_seq_id_.erase(state.scheduling->seq_id);
        running_.erase(running_it);
        skip_.push_front(request_it);
        co_return;
    }
}

void Scheduler::fail_all_waiting(ErrorCode err) {
    auto fail_one = [&](const RequestPtr& request_it) {
        if (request_it->scheduling) {
            release_scheduling_blocks(*request_it);
        }
        send_event(request_it->sink, std::unexpected(err));
        erase_request(request_it);
    };

    while (!waiting_.empty()) {
        auto request_it = waiting_.front();
        waiting_.pop_front();
        fail_one(request_it);
    }
    while (!skip_.empty()) {
        auto request_it = skip_.front();
        skip_.pop_front();
        fail_one(request_it);
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
        if (state.status == RequestStatus::Active) send_error_event(state, shutdown_err);

        release_scheduling_blocks(state);
        running_.erase(running_it);
        erase_request(request_it);
    }
    co_return;
}

asio::awaitable<void> Scheduler::wait_for_work() {
    if (!accepting_.load() || !waiting_.empty() || has_schedulable_work()) co_return;
    idle_timer_.expires_after(std::chrono::hours(24));
    auto [ec] = co_await idle_timer_.async_wait(as_tuple(deferred));
    (void)ec;
}

}  // namespace ccinfer
