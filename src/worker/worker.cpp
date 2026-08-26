#include "worker/worker.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "backend/backend.h"
#include "cache/kv_cache_manager.h"
#include "cache/kv_cache_storage.h"
#include "common/error_code.h"
#include "core/traits.h"
#include "facade/log.h"
#include "model/loader.h"
#include "worker/model_runner.h"
#include "model/registry.h"
#include "worker/batch_translator.h"

namespace ccinfer {

Worker::Worker(asio::io_context& io) : io_(io) {}

Worker::~Worker() { shutdown(); }

Result<void> Worker::init(const Config& config) {
    if (running_.load() || worker_thread_.joinable()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto r = init_resources(config);
    if (!r) {
        reset_resources();
        return r;
    }

    running_.store(true);
    worker_thread_ = std::thread(&Worker::worker_loop, this);
    return {};
}

void Worker::shutdown() {
    running_.store(false);
    cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

Result<AdmitSequenceResult> Worker::admit_sequence_resources(
    SequenceId seq_id, const std::vector<int32_t>& prompt_tokens, int max_context_len,
    SequenceInitialState initial_state) {
    std::lock_guard lock(resource_mutex_);
    if (!initialized_ || !backend_ || !kv_mgr_) {
        return std::unexpected(ErrorCode::ServerShuttingDown);
    }
    if (seq_id == 0 || prompt_tokens.empty() || max_context_len <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (initial_state.last_token < -1 || initial_state.tokens_generated < 0 ||
        initial_state.max_tokens <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (prompt_tokens.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (static_cast<int64_t>(max_context_len) < static_cast<int64_t>(prompt_tokens.size())) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (static_cast<int64_t>(max_context_len) >
        static_cast<int64_t>(max_blocks_.load()) * engine_config_.block_size) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (model_ && max_context_len > model_->config().max_seq_len_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    const int vocab_size = model_ ? model_->config().vocab_size_ : 0;
    for (auto tok : prompt_tokens) {
        if (tok < 0 || (vocab_size > 0 && tok >= vocab_size))
            return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (active_sequences_.load() >= engine_config_.max_sequences) {
        return std::unexpected(ErrorCode::MaxSequencesReached);
    }
    if (request_states_.count(seq_id) > 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    auto pr = kv_mgr_->lookup_prefix_cache(prompt_tokens, /*namespace_salt=*/0);
    if (!pr) {
        sync_capacity();
        return std::unexpected(pr.error());
    }

    int64_t prefix64 = static_cast<int64_t>(pr->prefix_hit_blocks) * kv_mgr_->block_size();
    int prefix_tokens =
        static_cast<int>(std::min(prefix64, static_cast<int64_t>(prompt_tokens.size())));

    SequenceState state;
    state.seq_id = seq_id;
    state.prompt_len = static_cast<int>(prompt_tokens.size());
    state.max_context_len = max_context_len;
    state.kv_written = prefix_tokens;
    state.prompt_processed = prefix_tokens;
    state.block_table = std::move(pr->block_table);
    state.parent_hash = pr->parent_hash;
    state.pending_tokens = std::move(pr->pending_tokens);
    state.last_token = initial_state.last_token;
    state.tokens_generated = initial_state.tokens_generated;
    state.max_tokens = initial_state.max_tokens;
    request_states_[seq_id] = std::move(state);
    active_sequences_++;
    sync_capacity();
    return AdmitSequenceResult{.seq_id = seq_id, .prompt_processed = prefix_tokens};
}

Result<void> Worker::release_sequence_resources(SequenceId seq_id) {
    std::lock_guard lock(resource_mutex_);
    if (!initialized_) return {};
    auto it = request_states_.find(seq_id);
    if (it == request_states_.end()) return {};
    if (!it->second.block_table.empty()) {
        auto r = kv_mgr_->release_blocks(it->second.block_table);
        if (!r) return r;
    }
    active_sequences_--;
    request_states_.erase(it);
    sync_capacity();
    return {};
}

namespace {

template <typename Channel>
std::shared_ptr<Channel> make_result_channel(boost::asio::io_context& io) {
    return std::make_shared<Channel>(io.get_executor(), 1);
}

}  // namespace

Result<std::shared_ptr<AdmitSequenceChannel>> Worker::enqueue_admit_sequence(
    SequenceId seq_id, std::vector<int32_t> prompt_tokens, int max_context_len,
    SequenceInitialState initial_state) {
    auto channel = make_result_channel<AdmitSequenceChannel>(io_);
    std::unique_lock lock(queue_mutex_);
    if (!running_) return std::unexpected(ErrorCode::ServerShuttingDown);
    queue_.push_back(
        AdmitCommand{seq_id, std::move(prompt_tokens), max_context_len, initial_state, channel});
    lock.unlock();
    cv_.notify_one();
    return channel;
}

Result<std::shared_ptr<VoidChannel>> Worker::enqueue_release_sequence(SequenceId seq_id) {
    auto channel = make_result_channel<VoidChannel>(io_);
    std::unique_lock lock(queue_mutex_);
    if (!running_) return std::unexpected(ErrorCode::ServerShuttingDown);
    queue_.push_back(ReleaseCommand{seq_id, channel});
    lock.unlock();
    cv_.notify_one();
    return channel;
}

Result<BatchFuture> Worker::enqueue_execute_batch(ScheduledBatch batch) {
    auto channel = make_result_channel<BatchChannel>(io_);
    std::unique_lock lock(queue_mutex_);
    if (!running_) return std::unexpected(ErrorCode::ServerShuttingDown);
    queue_.push_back(PendingBatch{std::move(batch), channel});
    lock.unlock();
    cv_.notify_one();
    return channel;
}

Capacity Worker::capacity() const {
    return Capacity{
        engine_config_.max_sequences, active_sequences_.load(),    free_blocks_.load(),
        max_blocks_.load(),           block_size_.load(),          block_active_.load(),
        block_cached_idle_.load(),    prefix_lookup_hits_.load(),  prefix_lookup_misses_.load(),
        prefix_evictions_.load(),     prefix_cached_blocks_.load()};
}

void Worker::worker_loop() {
    while (true) {
        std::deque<PendingCommand> local_queue;
        {
            std::unique_lock lock(queue_mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || !running_; });

            if (!running_ && queue_.empty()) {
                break;
            }

            local_queue = std::move(queue_);
            queue_.clear();
        }

        for (auto& command : local_queue) process_command(std::move(command));
    }

    reset_resources();
}

void Worker::process_command(PendingCommand command) {
    if (!running_.load()) {
        std::visit(
            [this](auto&& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, AdmitCommand>) {
                    resolve(value.channel, Result<AdmitSequenceResult>(
                                               std::unexpected(ErrorCode::ServerShuttingDown)));
                } else if constexpr (std::is_same_v<T, ReleaseCommand>) {
                    resolve(value.channel,
                            Result<void>(std::unexpected(ErrorCode::ServerShuttingDown)));
                } else {
                    resolve(value.chan, Result<WorkerBatchResult>(
                                            std::unexpected(ErrorCode::ServerShuttingDown)));
                }
            },
            std::move(command));
        return;
    }

    std::visit(
        [this](auto&& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, AdmitCommand>) {
                process_admit(std::move(value));
            } else if constexpr (std::is_same_v<T, ReleaseCommand>) {
                process_release(std::move(value));
            } else {
                process_batch(std::move(value));
            }
        },
        std::move(command));
}

void Worker::process_admit(AdmitCommand command) {
    resolve(command.channel,
            admit_sequence_resources(command.seq_id, command.prompt_tokens, command.max_context_len,
                                     command.initial_state));
}

void Worker::process_release(ReleaseCommand command) {
    resolve(command.channel, release_sequence_resources(command.seq_id));
}

void Worker::sync_capacity() {
    if (!kv_mgr_) return;
    auto s = kv_mgr_->stats();
    free_blocks_.store(s.block_free);
    max_blocks_.store(s.block_total);
    block_size_.store(s.block_size);
    block_active_.store(s.block_active);
    block_cached_idle_.store(s.block_cached_idle);
    prefix_lookup_hits_.store(s.prefix.lookup_hits);
    prefix_lookup_misses_.store(s.prefix.lookup_misses);
    prefix_evictions_.store(s.prefix.evictions);
    prefix_cached_blocks_.store(static_cast<uint64_t>(s.prefix.cached_blocks));
}

Result<void> Worker::init_resources(const Config& config) {
    std::lock_guard lock(resource_mutex_);
    try {
        engine_config_ = config.engine_;

        auto b = Backend::create(engine_config_.device_id);
        if (!b) {
            return std::unexpected(b.error());
        }
        backend_ = std::move(*b);

        static std::once_flag register_flag;
        std::call_once(register_flag, register_builtin_models);

        auto loader_r = WeightLoader::create(config.model_path_ + "/model.safetensors");
        if (!loader_r) {
            return std::unexpected(loader_r.error());
        }

        auto model_r = ModelRegistry::instance().create(config.model_, *loader_r, *backend_);
        if (!model_r) {
            return std::unexpected(model_r.error());
        }
        model_ = std::move(*model_r);

        const int num_layers = config.model_.n_layers_;
        const int num_kv_heads = config.model_.n_kv_heads_;
        const int head_dim = config.model_.head_dim_;

        max_blocks_.store(engine_config_.max_blocks);

        auto kvs_r = KVCacheStorage::create(*backend_, num_layers, max_blocks_.load(),
                                            engine_config_.block_size, num_kv_heads, head_dim,
                                            ccop::DType::kBFloat16);
        if (!kvs_r) {
            return std::unexpected(kvs_r.error());
        }

        kv_mgr_ = std::make_unique<KVCacheManager>();
        auto kmgr_r =
            kv_mgr_->init(std::move(*kvs_r), max_blocks_.load(), engine_config_.block_size);
        if (!kmgr_r) {
            return std::unexpected(kmgr_r.error());
        }

        sync_capacity();
        initialized_ = true;
        return {};
    } catch (const std::exception&) {
        return std::unexpected(ErrorCode::InternalError);
    } catch (...) {
        return std::unexpected(ErrorCode::InternalError);
    }
}

void Worker::reset_resources() {
    std::lock_guard lock(resource_mutex_);
    initialized_ = false;
    active_sequences_.store(0);
    free_blocks_.store(0);
    max_blocks_.store(0);
    block_size_.store(0);
    block_active_.store(0);
    block_cached_idle_.store(0);
    prefix_lookup_hits_.store(0);
    prefix_lookup_misses_.store(0);
    prefix_evictions_.store(0);
    prefix_cached_blocks_.store(0);
    request_states_.clear();
    model_.reset();
    kv_mgr_.reset();
    backend_.reset();
}

Worker::SequenceRegistry Worker::build_sequence_states(const ScheduledBatch& batch) const {
    SequenceRegistry sequences;
    sequences.reserve(batch.items.size());
    for (const auto& item : batch.items) {
        const SequenceId seq_id = work_sequence_id(item);
        if (sequences.count(seq_id) > 0) continue;
        auto it = request_states_.find(seq_id);
        if (it != request_states_.end()) sequences.emplace(seq_id, it->second);
    }
    return sequences;
}

Result<Worker::ResolvedBatch> Worker::resolve_batch(const ScheduledBatch& batch) const {
    ResolvedBatch resolved;
    resolved.batch.batch_id = batch.batch_id;
    resolved.batch.sampling = batch.sampling;
    resolved.batch.items.reserve(batch.items.size());
    resolved.original_indices.reserve(batch.items.size());

    for (std::size_t original_index = 0; original_index < batch.items.size(); ++original_index) {
        const auto& item = batch.items[original_index];
        const SequenceId seq_id = work_sequence_id(item);
        auto state_it = request_states_.find(seq_id);
        if (state_it == request_states_.end()) {
            WorkItemResult stale_result;
            stale_result.item_index = static_cast<int>(original_index);
            stale_result.seq_id = seq_id;
            stale_result.kind = work_kind(item);
            stale_result.stale = true;
            resolved.stale_results.push_back(std::move(stale_result));
            continue;
        }
        const auto& state = state_it->second;

        bool stale = false;
        WorkItem resolved_item = item;
        std::visit(
            [&](auto& work) {
                using T = std::decay_t<decltype(work)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    stale = state.finished || state.prompt_processed != work.prompt_span.start;
                    if (!stale) work.expected_context_len = state.kv_written;
                } else {
                    stale = state.finished ||
                            state.prompt_processed != state.prompt_len;
                    if (!stale && work.late_bind) {
                        stale = state.last_token < 0;
                        if (!stale) {
                            work.input_token = state.last_token;
                            work.expected_context_len = state.kv_written;
                            work.late_bind = false;
                        }
                    }
                }
            },
            resolved_item);

        if (stale) {
            WorkItemResult stale_result;
            stale_result.item_index = static_cast<int>(original_index);
            stale_result.seq_id = seq_id;
            stale_result.kind = work_kind(item);
            stale_result.stale = true;
            resolved.stale_results.push_back(std::move(stale_result));
            continue;
        }

        resolved.batch.items.push_back(std::move(resolved_item));
        resolved.original_indices.push_back(original_index);
    }

    return resolved;
}

Result<std::vector<SequenceDelta>> Worker::build_deltas(const SequenceRegistry& before,
                                                        const SequenceRegistry& after) {
    std::vector<SequenceDelta> deltas;
    deltas.reserve(after.size());
    for (const auto& [seq_id, state] : after) {
        auto before_it = before.find(seq_id);
        if (before_it == before.end() || request_states_.find(seq_id) == request_states_.end()) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        SequenceDelta delta;
        delta.seq_id = seq_id;
        delta.kv_tokens_committed = state.kv_written - before_it->second.kv_written;
        delta.prompt_tokens_committed = state.prompt_processed - before_it->second.prompt_processed;
        if (delta.kv_tokens_committed < 0 || delta.prompt_tokens_committed < 0) {
            return std::unexpected(ErrorCode::InternalError);
        }
        if (delta.kv_tokens_committed > 0 || delta.prompt_tokens_committed > 0) {
            deltas.push_back(delta);
        }
    }
    for (const auto& [seq_id, state] : after) request_states_.at(seq_id) = state;
    return deltas;
}

Result<void> Worker::apply_sampled_progress(SequenceRegistry& states, const BatchResult& batch) {
    for (const auto& work_result : batch.items) {
        auto it = states.find(work_result.seq_id);
        if (it == states.end()) return std::unexpected(ErrorCode::InvalidArgument);
        auto& state = it->second;
        if (!work_result.sampled_tokens.empty()) {
            state.last_token = work_result.sampled_tokens.front();
            ++state.tokens_generated;
        }
        state.finished =
            work_result.eos || (state.max_tokens > 0 && state.tokens_generated >= state.max_tokens);
    }
    return {};
}

void Worker::process_batch(PendingBatch pending) {
    std::lock_guard resource_lock(resource_mutex_);

    if (!initialized_ || !backend_ || !kv_mgr_) {
        resolve(pending.chan,
                Result<WorkerBatchResult>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }

    if (pending.batch.items.empty()) {
        resolve(pending.chan,
                Result<WorkerBatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
        return;
    }

    auto resolved_r = resolve_batch(pending.batch);
    if (!resolved_r) {
        resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(resolved_r.error())));
        return;
    }
    auto resolved = std::move(*resolved_r);
    if (resolved.batch.items.empty()) {
        WorkerBatchResult worker_result;
        worker_result.batch.batch_id = pending.batch.batch_id;
        worker_result.batch.items = std::move(resolved.stale_results);
        resolve(pending.chan, Result<WorkerBatchResult>(std::move(worker_result)));
        return;
    }

    auto& execution_batch = resolved.batch;

    auto before = build_sequence_states(execution_batch);
    if (before.size() != execution_batch.items.size()) {
        resolve(pending.chan,
                Result<WorkerBatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
        return;
    }
    auto sequences = before;

    assert(model_ && "Worker initialized without a model");
    BatchTranslator translator(*backend_, *kv_mgr_, engine_config_.block_size);

    auto plan_r = translator.prepare(execution_batch, sequences);
    if (!plan_r) {
        sync_capacity();  // translate may have evicted LRU blocks.
        resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(plan_r.error())));
        return;
    }
    auto plan = std::move(*plan_r);

    sync_capacity();  // reflect allocated blocks before forward.

    BatchResult result;
    result.batch_id = pending.batch.batch_id;

    if (plan.physical_batch.num_tokens > 0) {
        auto& phys_batch = plan.physical_batch;
        auto exec_r = ModelRunner::inference<BF16RunnerTraits>(*model_, phys_batch, *backend_,
                                                               *kv_mgr_, execution_batch.sampling);
        if (!exec_r) {
            plan.rollback();
            sync_capacity();
            resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(exec_r.error())));
            return;
        }
        result.items = std::move(*exec_r);
        for (auto& wr : result.items) {
            if (wr.item_index >= 0 &&
                static_cast<std::size_t>(wr.item_index) < plan.no_write_flags.size() &&
                plan.no_write_flags[static_cast<std::size_t>(wr.item_index)]) {
                wr.kv_deferred = true;
            }
        }
    }

    for (const std::size_t deferred_index : plan.deferred_indices) {
        WorkItemResult wr;
        wr.item_index = static_cast<int>(deferred_index);
        wr.seq_id = work_sequence_id(execution_batch.items[deferred_index]);
        wr.kind = work_kind(execution_batch.items[deferred_index]);
        wr.tokens_consumed = 0;
        wr.kv_deferred = true;
        result.items.push_back(std::move(wr));
    }

    for (auto& wr : result.items) {
        if (wr.item_index < 0 ||
            static_cast<std::size_t>(wr.item_index) >= execution_batch.items.size()) {
            plan.rollback();
            sync_capacity();
            resolve(pending.chan,
                    Result<WorkerBatchResult>(std::unexpected(ErrorCode::BatchTranslationFailed)));
            return;
        }

        const auto& requested_item = execution_batch.items[static_cast<std::size_t>(wr.item_index)];
        if (wr.kind != work_kind(requested_item) ||
            (!wr.kv_deferred && wr.tokens_consumed != work_token_count(requested_item))) {
            plan.rollback();
            sync_capacity();
            resolve(pending.chan,
                    Result<WorkerBatchResult>(std::unexpected(ErrorCode::BatchTranslationFailed)));
            return;
        }
    }

    auto commit_r = plan.commit();
    if (!commit_r) {
        plan.rollback();
        sync_capacity();
        resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(commit_r.error())));
        return;
    }

    // Cache full blocks into prefix cache for every item that actually wrote KV.
    const int block_size = kv_mgr_->block_size();
    for (std::size_t i = 0; i < execution_batch.items.size(); ++i) {
        if (std::find(plan.deferred_indices.begin(), plan.deferred_indices.end(), i) !=
            plan.deferred_indices.end()) {
            continue;
        }
        if (i < plan.no_write_flags.size() && plan.no_write_flags[i]) {
            continue;
        }

        const auto& item = execution_batch.items[i];
        const SequenceId seq_id = work_sequence_id(item);
        auto it = sequences.find(seq_id);
        if (it == sequences.end()) continue;
        auto& seq = it->second;

        std::vector<int32_t> written_tokens;
        if (const auto* pc = std::get_if<PrefillChunk>(&item)) {
            written_tokens = pc->tokens;
        } else if (const auto* d = std::get_if<DecodeOneToken>(&item)) {
            if (d->write_kv) written_tokens.push_back(d->input_token);
        }

        const int write_start = seq.kv_written - static_cast<int>(written_tokens.size());
        std::vector<int32_t> block_ids;
        for (std::size_t j = 0; j < written_tokens.size(); ++j) {
            seq.pending_tokens.push_back(written_tokens[j]);
            if (seq.pending_tokens.size() % static_cast<std::size_t>(block_size) == 0) {
                const int global_pos = write_start + static_cast<int>(j);
                const int block_index = global_pos / block_size;
                block_ids.push_back(seq.block_table[static_cast<std::size_t>(block_index)]);
            }
        }

        if (!block_ids.empty()) {
            auto new_hash_r =
                kv_mgr_->cache_rolling_blocks(seq.parent_hash, seq.pending_tokens, block_ids);
            if (!new_hash_r) {
                ccLog::error("cache_rolling_blocks failed seq={} blocks={} err={}", seq.seq_id,
                             block_ids.size(), static_cast<int>(new_hash_r.error()));
                continue;
            }
            seq.parent_hash = *new_hash_r;
            seq.pending_tokens.erase(
                seq.pending_tokens.begin(),
                seq.pending_tokens.begin() +
                    static_cast<std::ptrdiff_t>(block_ids.size() * static_cast<std::size_t>(block_size)));
        }
    }

    sync_capacity();

    auto progress_r = apply_sampled_progress(sequences, result);
    if (!progress_r) {
        resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(progress_r.error())));
        return;
    }

    auto deltas_r = build_deltas(before, sequences);
    if (!deltas_r) {
        resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(deltas_r.error())));
        return;
    }
    auto deltas = std::move(*deltas_r);

    WorkerBatchResult worker_result;
    worker_result.batch = std::move(result);
    for (auto& wr : worker_result.batch.items) {
        wr.item_index = static_cast<int>(
            resolved.original_indices[static_cast<std::size_t>(wr.item_index)]);
    }
    worker_result.batch.items.insert(worker_result.batch.items.end(),
                                     std::make_move_iterator(resolved.stale_results.begin()),
                                     std::make_move_iterator(resolved.stale_results.end()));
    worker_result.deltas = std::move(deltas);
    std::sort(worker_result.batch.items.begin(), worker_result.batch.items.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.item_index < rhs.item_index; });
    resolve(pending.chan, Result<WorkerBatchResult>(std::move(worker_result)));
}

}  // namespace ccinfer
