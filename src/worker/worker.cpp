#include "worker/worker.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <cuda_bf16.h>

#include "backend/backend.h"
#include "base/error_code.h"
#include "cache/kv_cache_manager.h"
#include "cache/kv_cache_storage.h"
#include "core/traits.h"
#include "facade/log.h"
#include "model/loader.h"
#include "model/model_runner.h"
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

Result<CreateSequenceResult> Worker::prepare_sequence_resources(
    SequenceId seq_id, const std::vector<int32_t>& prompt_tokens, int max_context_len) {
    std::lock_guard lock(resource_mutex_);
    if (!initialized_ || !backend_ || !kv_mgr_) {
        return std::unexpected(ErrorCode::ServerShuttingDown);
    }
    if (seq_id == 0 || prompt_tokens.empty() || max_context_len <= 0) {
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
    for (auto tok : prompt_tokens) {
        if (tok < 0) return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (active_sequences_.load() >= engine_config_.max_sequences) {
        return std::unexpected(ErrorCode::MaxSequencesReached);
    }
    if (block_tables_.count(seq_id) > 0) {
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

    block_tables_[seq_id] = std::move(pr->block_table);
    active_sequences_++;
    sync_capacity();
    return CreateSequenceResult{.seq_id = seq_id, .prompt_processed = prefix_tokens};
}

Result<SuspendSequenceResult> Worker::reset_sequence_resources(
    SequenceId seq_id, const std::vector<int32_t>& prompt_tokens, int max_context_len) {
    std::lock_guard lock(resource_mutex_);
    if (!initialized_ || !backend_ || !kv_mgr_) {
        return std::unexpected(ErrorCode::ServerShuttingDown);
    }
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (prompt_tokens.empty() || max_context_len <= 0 ||
        prompt_tokens.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        static_cast<int64_t>(max_context_len) < static_cast<int64_t>(prompt_tokens.size()) ||
        static_cast<int64_t>(max_context_len) >
            static_cast<int64_t>(max_blocks_.load()) * engine_config_.block_size) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    for (auto tok : prompt_tokens) {
        if (tok < 0) return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (!it->second.empty()) {
        auto r = kv_mgr_->release_blocks(it->second);
        if (!r) {
            sync_capacity();
            return std::unexpected(r.error());
        }
        it->second = {};
    }

    auto pr = kv_mgr_->lookup_prefix_cache(prompt_tokens, /*namespace_salt=*/0);
    if (!pr) {
        sync_capacity();
        return std::unexpected(pr.error());
    }

    int64_t prefix64 = static_cast<int64_t>(pr->prefix_hit_blocks) * kv_mgr_->block_size();
    int prefix_tokens =
        static_cast<int>(std::min(prefix64, static_cast<int64_t>(prompt_tokens.size())));
    it->second = std::move(pr->block_table);

    sync_capacity();
    return SuspendSequenceResult{.prompt_processed = prefix_tokens};
}

Result<void> Worker::release_sequence_resources(SequenceId seq_id) {
    std::lock_guard lock(resource_mutex_);
    if (!initialized_) return {};
    auto it = block_tables_.find(seq_id);
    if (it == block_tables_.end()) return {};
    return release_sequence_blocks(it);
}

void Worker::enqueue_execute_batch(ScheduledBatch batch, std::vector<SequenceSnapshot> sequences,
                                   std::shared_ptr<BatchChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, Result<WorkerBatchResult>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }
    queue_.push_back(PendingBatch{std::move(batch), std::move(sequences), std::move(chan)});
    lock.unlock();
    cv_.notify_one();
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
        std::deque<PendingBatch> local_queue;
        {
            std::unique_lock lock(queue_mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || !running_; });

            if (!running_ && queue_.empty()) {
                break;
            }

            local_queue = std::move(queue_);
            queue_.clear();
        }

        for (auto& entry : local_queue) {
            if (!running_) {
                resolve(entry.chan,
                        Result<WorkerBatchResult>(std::unexpected(ErrorCode::ServerShuttingDown)));
                continue;
            }

            process_batch(std::move(entry));
        }
    }

    reset_resources();
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
        auto config_r = config.engine_.validate();
        if (!config_r) return std::unexpected(config_r.error());
        engine_config_ = config.engine_;

        auto b = Backend::create(engine_config_.device_id);
        if (!b) {
            return std::unexpected(b.error());
        }
        backend_ = std::move(*b);

        static std::once_flag register_flag;
        std::call_once(register_flag, register_builtin_models);

        int num_layers = 0;
        int num_kv_heads = 0;
        int head_dim = 0;

        if (!config.model_path_.empty()) {
            num_layers = config.model_.n_layers_;
            num_kv_heads = config.model_.n_kv_heads_;
            head_dim = config.model_.head_dim_;

            auto loader_r = WeightLoader::create(config.model_path_ + "/model.safetensors");
            if (!loader_r) {
                return std::unexpected(loader_r.error());
            }

            auto model_r = ModelRegistry::instance().create(config.model_, *loader_r, *backend_);
            if (!model_r) {
                return std::unexpected(model_r.error());
            }
            model_ = std::move(*model_r);
        } else {
            num_layers = engine_config_.dummy_num_layers;
            num_kv_heads = engine_config_.dummy_num_kv_heads;
            head_dim = engine_config_.dummy_head_dim;
        }

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
    block_tables_.clear();
    model_.reset();
    kv_mgr_.reset();
    backend_.reset();
}

Result<void> Worker::release_sequence_blocks(BlockTableIter it) {
    if (!it->second.empty()) {
        auto r = kv_mgr_->release_blocks(it->second);
        if (!r) return r;
    }
    active_sequences_--;
    block_tables_.erase(it);
    sync_capacity();
    return {};
}

std::unordered_map<SequenceId, SequenceState> Worker::build_sequence_states(
    const std::unordered_map<SequenceId, SequenceSnapshot>& snapshots) const {
    std::unordered_map<SequenceId, SequenceState> sequences;
    sequences.reserve(snapshots.size());
    for (const auto& [seq_id, snapshot] : snapshots) {
        SequenceState state;
        state.seq_id = snapshot.seq_id;
        state.prompt_tokens = snapshot.prompt_tokens;
        state.max_context_len = snapshot.max_context_len;
        state.kv_written = snapshot.kv_written;
        state.prompt_processed = snapshot.prompt_processed;
        state.block_table = block_tables_.at(seq_id);
        state.status = snapshot.status;
        sequences.emplace(seq_id, std::move(state));
    }
    return sequences;
}

Result<std::vector<SequenceDelta>> Worker::build_deltas(
    const std::unordered_map<SequenceId, SequenceState>& sequences,
    const std::unordered_map<SequenceId, SequenceSnapshot>& snapshots) {
    std::vector<SequenceDelta> deltas;
    deltas.reserve(sequences.size());
    for (const auto& [seq_id, state] : sequences) {
        const auto& before = snapshots.at(seq_id);
        auto bt = block_tables_.find(seq_id);
        if (bt == block_tables_.end()) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        bt->second = state.block_table;
        SequenceDelta delta;
        delta.seq_id = seq_id;
        delta.kv_tokens_committed = state.kv_written - before.kv_written;
        delta.prompt_tokens_committed = state.prompt_processed - before.prompt_processed;
        if (delta.kv_tokens_committed < 0 || delta.prompt_tokens_committed < 0) {
            return std::unexpected(ErrorCode::InternalError);
        }
        if (delta.kv_tokens_committed > 0 || delta.prompt_tokens_committed > 0) {
            deltas.push_back(delta);
        }
    }
    return deltas;
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

    std::unordered_map<SequenceId, SequenceSnapshot> snapshots;
    for (auto& snapshot : pending.sequences) {
        if (snapshot.seq_id == 0 || snapshots.count(snapshot.seq_id) > 0) {
            resolve(pending.chan,
                    Result<WorkerBatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
            return;
        }
        if (block_tables_.find(snapshot.seq_id) == block_tables_.end()) {
            resolve(pending.chan,
                    Result<WorkerBatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
            return;
        }
        snapshots.emplace(snapshot.seq_id, std::move(snapshot));
    }

    // Validate all seq_ids exist in both logical snapshots and device block tables.
    for (const auto& item : pending.batch.items) {
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        if (snapshots.find(seq_id) == snapshots.end() ||
            block_tables_.find(seq_id) == block_tables_.end()) {
            resolve(pending.chan,
                    Result<WorkerBatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
            return;
        }
    }

    auto sequences = build_sequence_states(snapshots);

    if (model_) {
        BatchTranslator translator(*backend_, *kv_mgr_, engine_config_.block_size);

        auto plan_r = translator.prepare(pending.batch, sequences);
        if (!plan_r) {
            sync_capacity();  // translate may have evicted LRU blocks.
            resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(plan_r.error())));
            return;
        }
        auto plan = std::move(*plan_r);

        sync_capacity();  // reflect allocated blocks before forward.

        auto& phys_batch = plan.physical_batch;

        auto exec_r = ModelRunner::inference<BF16RunnerTraits>(*model_, phys_batch, *backend_,
                                                               *kv_mgr_, pending.batch.sampling);

        if (!exec_r) {
            plan.rollback();
            sync_capacity();
            resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(exec_r.error())));
            return;
        }

        BatchResult result;
        result.batch_id = pending.batch.batch_id;
        result.items = std::move(*exec_r);
        for (auto& wr : result.items) {
            if (wr.item_index < 0 ||
                static_cast<std::size_t>(wr.item_index) >= pending.batch.items.size()) {
                plan.rollback();
                sync_capacity();
                resolve(pending.chan, Result<WorkerBatchResult>(
                                          std::unexpected(ErrorCode::BatchTranslationFailed)));
                return;
            }

            const auto& requested_item =
                pending.batch.items[static_cast<std::size_t>(wr.item_index)];
            WorkKind expected_kind = WorkKind::PrefillChunk;
            int expected_tokens = 0;
            std::visit(
                [&](const auto& w) {
                    using T = std::decay_t<decltype(w)>;
                    if constexpr (std::is_same_v<T, PrefillChunk>) {
                        expected_kind = WorkKind::PrefillChunk;
                        expected_tokens = w.prompt_span.length;
                    } else {
                        expected_kind = WorkKind::DecodeOneToken;
                        expected_tokens = 1;
                    }
                },
                requested_item);
            if (wr.kind != expected_kind || wr.tokens_consumed != expected_tokens) {
                plan.rollback();
                sync_capacity();
                resolve(pending.chan, Result<WorkerBatchResult>(
                                          std::unexpected(ErrorCode::BatchTranslationFailed)));
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

        // Cache full blocks into prefix cache for prefill items.
        for (const auto& item : pending.batch.items) {
            if (std::holds_alternative<PrefillChunk>(item)) {
                const auto& pc = std::get<PrefillChunk>(item);
                auto it = sequences.find(pc.seq_id);
                if (it != sequences.end()) {
                    const auto& seq = it->second;
                    auto cf_r = kv_mgr_->cache_full_blocks(seq.block_table, seq.prompt_tokens,
                                                           seq.kv_written,
                                                           /*namespace_salt=*/0);
                    if (!cf_r) {
                        ccLog::error("cache_full_blocks failed seq={} err={}", seq.seq_id,
                                     static_cast<int>(cf_r.error()));
                    }
                }
            }
        }

        sync_capacity();

        auto deltas_r = build_deltas(sequences, snapshots);
        if (!deltas_r) {
            resolve(pending.chan, Result<WorkerBatchResult>(std::unexpected(deltas_r.error())));
            return;
        }
        auto deltas = std::move(*deltas_r);

        WorkerBatchResult worker_result;
        worker_result.batch = std::move(result);
        worker_result.deltas = std::move(deltas);
        resolve(pending.chan, Result<WorkerBatchResult>(std::move(worker_result)));
        return;
    }

    // Dummy mode (no model loaded) keeps the same batch-result and logical
    // progress contract as the real path for control-path tests.
    auto worker_result = generate_dummy_batch_result(pending.batch);
    resolve(pending.chan, Result<WorkerBatchResult>(std::move(worker_result)));
}

WorkerBatchResult Worker::generate_dummy_batch_result(const ScheduledBatch& batch) const {
    WorkerBatchResult worker_result;
    worker_result.batch.batch_id = batch.batch_id;

    for (std::size_t i = 0; i < batch.items.size(); ++i) {
        WorkItemResult wr;
        wr.item_index = static_cast<int>(i);
        wr.sampled_tokens = {42};

        std::visit(
            [&](const auto& w) {
                wr.seq_id = w.seq_id;
                using W = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<W, PrefillChunk>) {
                    wr.kind = WorkKind::PrefillChunk;
                    wr.tokens_consumed = w.prompt_span.length;
                } else {
                    wr.kind = WorkKind::DecodeOneToken;
                    wr.tokens_consumed = 1;
                }
            },
            batch.items[i]);

        worker_result.batch.items.push_back(std::move(wr));

        SequenceDelta delta;
        std::visit(
            [&](const auto& w) {
                delta.seq_id = w.seq_id;
                using T = std::decay_t<decltype(w)>;
                if constexpr (std::is_same_v<T, PrefillChunk>) {
                    delta.kv_tokens_committed = w.prompt_span.length;
                    delta.prompt_tokens_committed = w.prompt_span.length;
                } else {
                    delta.kv_tokens_committed = w.write_kv ? 1 : 0;
                }
            },
            batch.items[i]);
        worker_result.deltas.push_back(delta);
    }

    return worker_result;
}

}  // namespace ccinfer
