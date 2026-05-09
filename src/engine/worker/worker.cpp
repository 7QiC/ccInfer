#include "engine/worker/worker.h"

#include <cuda_bf16.h>

#include <type_traits>

#include "engine/cache/kv_cache_manager.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/common/backend_def.h"
#include "engine/common/traits.h"
#include "engine/model/model_runner.h"

namespace ccinfer {
namespace engine {

Worker::Worker(asio::io_context& io) : io_(io) {}

Worker::~Worker() { shutdown(); }

Result<void> Worker::init(const std::string& model_path) {
    running_ = true;
    worker_thread_ = std::thread(&Worker::worker_loop, this);

    // Post init to worker thread so CUDA resources bind to the worker's thread.
    auto promise = std::make_shared<std::promise<Result<void>>>();
    auto future = promise->get_future();
    {
        std::lock_guard lock(queue_mutex_);
        queue_.push_back(ControlCmd{CmdInitResources{model_path, promise}});
    }
    cv_.notify_one();

    auto result = future.get();
    if (!result) shutdown();
    return result;
}

void Worker::shutdown() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

// --- Enqueue helpers ---

void Worker::enqueue_create_sequence(std::vector<int32_t> prompt_tokens,
                                     int max_context_len,
                                     std::shared_ptr<SeqIdChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, ErrorCode::ServerShuttingDown, SequenceId{0});
        return;
    }
    queue_.push_back(
        ControlCmd{CmdCreateSequence{std::move(prompt_tokens), max_context_len, std::move(chan)}});
    lock.unlock();
    cv_.notify_one();
}

void Worker::enqueue_release_sequence(SequenceId seq_id,
                                      std::shared_ptr<VoidChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, ErrorCode::ServerShuttingDown);
        return;
    }
    queue_.push_back(ControlCmd{CmdReleaseSequence{seq_id, std::move(chan)}});
    lock.unlock();
    cv_.notify_one();
}

void Worker::enqueue_abort_sequence(SequenceId seq_id,
                                    std::shared_ptr<VoidChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, ErrorCode::ServerShuttingDown);
        return;
    }
    queue_.push_back(ControlCmd{CmdAbortSequence{seq_id, std::move(chan)}});
    lock.unlock();
    cv_.notify_one();
}

void Worker::enqueue_execute_batch(ScheduledBatch batch,
                                   std::shared_ptr<BatchChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        uint64_t batch_id = batch.batch_id;
        lock.unlock();
        BatchResult result;
        result.batch_id = batch_id;
        resolve(chan, ErrorCode::ServerShuttingDown, std::move(result));
        return;
    }
    queue_.push_back(PendingBatch{std::move(batch), std::move(chan)});
    lock.unlock();
    cv_.notify_one();
}

DeviceCapacity Worker::capacity() const {
    return DeviceCapacity{kMaxSequences, active_sequences_.load(), free_blocks_.load(),
                          max_blocks_};
}

// --- Worker loop ---

void Worker::worker_loop() {
    while (true) {
        std::deque<QueueEntry> local_queue;
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
            // If shutting down, resolve remaining entries with error and skip processing.
            if (!running_) {
                std::visit(
                    [this](auto& e) {
                        using T = std::decay_t<decltype(e)>;
                        if constexpr (std::is_same_v<T, ControlCmd>) {
                            std::visit(
                                [this](auto& c) {
                                    using CT = std::decay_t<decltype(c)>;
                                    if constexpr (std::is_same_v<CT, CmdInitResources>) {
                                        c.promise->set_value(
                                            std::unexpected(ErrorCode::ServerShuttingDown));
                                    } else if constexpr (std::is_same_v<CT, CmdCreateSequence>) {
                                        resolve(c.chan, ErrorCode::ServerShuttingDown,
                                                SequenceId{0});
                                    } else if constexpr (std::is_same_v<CT, CmdReleaseSequence> ||
                                                         std::is_same_v<CT, CmdAbortSequence>) {
                                        resolve(c.chan, ErrorCode::ServerShuttingDown);
                                    }
                                },
                                e);
                        } else if constexpr (std::is_same_v<T, PendingBatch>) {
                            BatchResult result;
                            result.batch_id = e.batch.batch_id;
                            resolve(e.chan, ErrorCode::ServerShuttingDown, std::move(result));
                        }
                    },
                    entry);
                continue;
            }

            std::visit(
                [this](auto& e) {
                    using T = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<T, ControlCmd>) {
                        process_control(e);
                    } else if constexpr (std::is_same_v<T, PendingBatch>) {
                        process_batch(std::move(e));
                    }
                },
                entry);
        }
    }

    // Cleanup resources on worker thread before exiting.
    model_.reset();
    kv_storage_.reset();
    kv_mgr_.reset();
    backend_.reset();
}

// --- Resource init (runs on worker thread) ---

void Worker::init_resources(const std::string& model_path,
                            std::shared_ptr<std::promise<Result<void>>> promise) {
    backend_ = std::make_unique<DefaultBackend>();
    auto r = backend_->init(0);
    if (!r) {
        promise->set_value(std::unexpected(r.error()));
        return;
    }

    // Model loading deferred — use dummy tokens via generate_dummy_result() for now.
    // Real loading (WeightLoader + ModelRegistry.create) wired later.

    constexpr int kDefaultLayers = 32;
    constexpr int kDefaultKVHeads = 8;
    constexpr int kDefaultHeadDim = 128;
    max_blocks_ = 1024;

    kv_mgr_ = std::make_unique<KVCacheManager>();
    auto kmgr_r = kv_mgr_->init(max_blocks_);
    if (!kmgr_r) {
        promise->set_value(std::unexpected(kmgr_r.error()));
        return;
    }

    kv_storage_ = std::make_unique<KVCacheStorage>();
    auto kvs_r = kv_storage_->init<__nv_bfloat16>(*backend_, kDefaultLayers, max_blocks_, kKVBlockSize,
                                   kDefaultKVHeads, kDefaultHeadDim);
    if (!kvs_r) {
        promise->set_value(std::unexpected(kvs_r.error()));
        return;
    }

    free_blocks_ = max_blocks_;
    promise->set_value({});
}

// --- Control processing ---

void Worker::process_control(ControlCmd& cmd) {
    std::visit(
        [this](auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, CmdInitResources>) {
                init_resources(std::move(c.model_path), c.promise);
            } else if constexpr (std::is_same_v<T, CmdCreateSequence>) {
                // Validate
                if (c.prompt_tokens.empty() || c.max_context_len <= 0) {
                    resolve(c.chan, ErrorCode::InvalidArgument, SequenceId{0});
                    return;
                }
                if (c.max_context_len < static_cast<int>(c.prompt_tokens.size())) {
                    resolve(c.chan, ErrorCode::InvalidArgument, SequenceId{0});
                    return;
                }
                if (c.max_context_len > max_blocks_ * kKVBlockSize) {
                    resolve(c.chan, ErrorCode::InvalidArgument, SequenceId{0});
                    return;
                }
                if (active_sequences_.load() >= kMaxSequences) {
                    resolve(c.chan, ErrorCode::MaxSequencesReached, SequenceId{0});
                    return;
                }

                SequenceId id = next_seq_id_++;
                SequenceState state;
                state.seq_id = id;
                state.prompt_tokens = std::move(c.prompt_tokens);
                state.max_context_len = c.max_context_len;
                sequences_[id] = std::move(state);
                active_sequences_++;
                resolve(c.chan, ErrorCode::Ok, id);
            } else if constexpr (std::is_same_v<T, CmdReleaseSequence>) {
                auto it = sequences_.find(c.seq_id);
                if (it == sequences_.end()) {
                    resolve(c.chan, ErrorCode::Ok);
                    return;
                }
                release_sequence_state(it);
                resolve(c.chan, ErrorCode::Ok);
            } else if constexpr (std::is_same_v<T, CmdAbortSequence>) {
                auto it = sequences_.find(c.seq_id);
                if (it == sequences_.end()) {
                    resolve(c.chan, ErrorCode::Ok);
                    return;
                }
                release_sequence_state(it);
                resolve(c.chan, ErrorCode::Ok);
            }
        },
        cmd);
}

// --- Sequence state helper ---

void Worker::release_sequence_state(SeqMapIter it) {
    int blocks = static_cast<int>(it->second.block_table.size());
    if (blocks > 0) {
        kv_mgr_->release_blocks(it->second.block_table);
        free_blocks_ += blocks;
    }
    active_sequences_--;
    sequences_.erase(it);
}

// --- Batch processing ---

void Worker::process_batch(PendingBatch pending) {
    // Validate all seq_ids exist before doing any work.
    for (const auto& item : pending.batch.items) {
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        if (sequences_.find(seq_id) == sequences_.end()) {
            BatchResult result;
            result.batch_id = pending.batch.batch_id;
            resolve(pending.chan, ErrorCode::InvalidArgument, std::move(result));
            return;
        }
    }

    BatchResult result;

    if (model_) {
        // model_ is loaded but BatchTranslator not yet wired (Task 10).
        // The fake PhysicalBatch below has null device buffers and would
        // cause undefined behaviour if passed to a real forward.
        // TODO: replace with BatchTranslator → PhysicalBatch → ModelRunner.
        result.batch_id = pending.batch.batch_id;
        resolve(pending.chan, ErrorCode::BatchTranslationFailed, std::move(result));
        return;
    }

    result = generate_dummy_result(pending.batch);
    resolve(pending.chan, ErrorCode::Ok, std::move(result));
}

BatchResult Worker::generate_dummy_result(const ScheduledBatch& batch) {
    // Phase 4.1 end-to-end wiring placeholder.
    //
    // Returns token 42 for every WorkItem so that the pipeline
    // HTTP → Scheduler → Engine → Worker → SSE can be exercised.
    //
    // Real behaviour (after BatchTranslator + real ModelRunner, Task 10–12):
    //   - PrefillChunk: typically writes KV cache only; only the *final*
    //     prefill chunk samples the first token.
    //   - DecodeOneToken: samples one next token per request.
    //   - EOS / max_tokens stop conditions are handled by the scheduler.

    BatchResult result;
    result.batch_id = batch.batch_id;

    for (size_t i = 0; i < batch.items.size(); ++i) {
        WorkItemResult wr;
        wr.item_index = static_cast<int>(i);
        wr.sampled_tokens = {42};

        std::visit(
            [&](const auto& w) {
                wr.seq_id = w.seq_id;
                if constexpr (std::is_same_v<std::decay_t<decltype(w)>, PrefillChunk>) {
                    wr.kind = WorkKind::PrefillChunk;
                    wr.tokens_consumed = w.prompt_span.length;
                } else {
                    wr.kind = WorkKind::DecodeOneToken;
                    wr.tokens_consumed = 1;
                }
            },
            batch.items[i]);

        result.items.push_back(std::move(wr));
    }

    return result;
}

}  // namespace engine
}  // namespace ccinfer
