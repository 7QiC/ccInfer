#include "engine/worker/worker.h"

#include <cuda_bf16.h>

#include <fstream>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>

#include "common/error_code.h"
#include "engine/cache/kv_cache_manager.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/common/backend_def.h"
#include "engine/common/traits.h"
#include "engine/model/config.h"
#include "engine/model/loader.h"
#include "engine/model/model_runner.h"
#include "engine/model/registry.h"
#include "engine/worker/batch_translator.h"

namespace ccinfer {
namespace engine {

Worker::Worker(asio::io_context& io) : io_(io) {}

Worker::~Worker() { shutdown(); }

Result<void> Worker::init(const std::string& model_path) {
    if (running_.load() || worker_thread_.joinable()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    running_.store(true);
    worker_thread_ = std::thread(&Worker::worker_loop, this);

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
    running_.store(false);
    cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

// --- Enqueue helpers ---

void Worker::enqueue_create_sequence(std::vector<int32_t> prompt_tokens, int max_context_len,
                                     std::shared_ptr<SeqIdChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, Result<SequenceId>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }
    queue_.push_back(
        ControlCmd{CmdCreateSequence{std::move(prompt_tokens), max_context_len, std::move(chan)}});
    lock.unlock();
    cv_.notify_one();
}

void Worker::enqueue_release_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, Result<void>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }
    queue_.push_back(ControlCmd{CmdReleaseSequence{seq_id, std::move(chan)}});
    lock.unlock();
    cv_.notify_one();
}

void Worker::enqueue_abort_sequence(SequenceId seq_id, std::shared_ptr<VoidChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, Result<void>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }
    queue_.push_back(ControlCmd{CmdAbortSequence{seq_id, std::move(chan)}});
    lock.unlock();
    cv_.notify_one();
}

void Worker::enqueue_execute_batch(ScheduledBatch batch, std::shared_ptr<BatchChannel> chan) {
    std::unique_lock lock(queue_mutex_);
    if (!running_) {
        lock.unlock();
        resolve(chan, Result<BatchResult>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }
    queue_.push_back(PendingBatch{std::move(batch), std::move(chan)});
    lock.unlock();
    cv_.notify_one();
}

DeviceCapacity Worker::capacity() const {
    return DeviceCapacity{kMaxSequences, active_sequences_.load(), free_blocks_.load(),
                          max_blocks_.load(), kv_storage_->block_size()};
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
                                        resolve(c.chan, Result<SequenceId>(std::unexpected(
                                                            ErrorCode::ServerShuttingDown)));
                                    } else if constexpr (std::is_same_v<CT, CmdReleaseSequence> ||
                                                         std::is_same_v<CT, CmdAbortSequence>) {
                                        resolve(c.chan, Result<void>(std::unexpected(
                                                            ErrorCode::ServerShuttingDown)));
                                    }
                                },
                                e);
                        } else if constexpr (std::is_same_v<T, PendingBatch>) {
                            resolve(e.chan, Result<BatchResult>(
                                                std::unexpected(ErrorCode::ServerShuttingDown)));
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

    initialized_ = false;
    active_sequences_.store(0);
    free_blocks_.store(0);
    max_blocks_.store(0);
    sequences_.clear();
    model_.reset();
    kv_storage_.reset();
    kv_mgr_.reset();
    backend_.reset();
}

// --- Resource init (runs on worker thread) ---

void Worker::init_resources(const std::string& model_path,
                            std::shared_ptr<std::promise<Result<void>>> promise) {
    try {
        auto b = DefaultBackend::create(0);
        if (!b) {
            promise->set_value(std::unexpected(b.error()));
            return;
        }
        backend_ = std::move(*b);

        static std::once_flag register_flag;
        std::call_once(register_flag, register_builtin_models);

        int num_layers = 0;
        int num_kv_heads = 0;
        int head_dim = 0;

        if (!model_path.empty()) {
            std::ifstream cfg_file(model_path + "/config.json");
            if (!cfg_file.is_open()) {
                promise->set_value(std::unexpected(ErrorCode::ModelLoadFailed));
                return;
            }
            auto cfg_j = nlohmann::json::parse(cfg_file, nullptr, false);
            if (cfg_j.is_discarded()) {
                promise->set_value(std::unexpected(ErrorCode::ModelConfigInvalid));
                return;
            }
            auto cfg_r = ModelConfig::from_json(cfg_j);
            if (!cfg_r) {
                promise->set_value(std::unexpected(cfg_r.error()));
                return;
            }
            num_layers = cfg_r->n_layers_;
            num_kv_heads = cfg_r->n_kv_heads_;
            head_dim = cfg_r->head_dim_;

            auto loader_r = WeightLoader::create(model_path + "/model.safetensors");
            if (!loader_r) {
                promise->set_value(std::unexpected(loader_r.error()));
                return;
            }

            auto model_r = ModelRegistry::instance().create(*cfg_r, *loader_r, *backend_);
            if (!model_r) {
                promise->set_value(std::unexpected(model_r.error()));
                return;
            }
            model_ = std::move(*model_r);
        } else {
            num_layers = 32;
            num_kv_heads = 8;
            head_dim = 128;
        }

        max_blocks_.store(1024);

        kv_mgr_ = std::make_unique<KVCacheManager>();
        auto kmgr_r = kv_mgr_->init(max_blocks_.load());
        if (!kmgr_r) {
            promise->set_value(std::unexpected(kmgr_r.error()));
            return;
        }

        kv_storage_ = std::make_unique<KVCacheStorage>();
        auto kvs_r = kv_storage_->init<__nv_bfloat16>(*backend_, num_layers, max_blocks_.load(),
                                                      kKVBlockSize, num_kv_heads, head_dim);
        if (!kvs_r) {
            promise->set_value(std::unexpected(kvs_r.error()));
            return;
        }

        free_blocks_.store(max_blocks_.load());
        initialized_ = true;
        promise->set_value({});
    } catch (const std::exception&) {
        promise->set_value(std::unexpected(ErrorCode::InternalError));
    } catch (...) {
        promise->set_value(std::unexpected(ErrorCode::InternalError));
    }
}

// --- Control processing ---

void Worker::process_control(ControlCmd& cmd) {
    std::visit(
        [this](auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, CmdInitResources>) {
                init_resources(std::move(c.model_path), c.promise);
            } else if constexpr (std::is_same_v<T, CmdCreateSequence>) {
                if (!initialized_ || !backend_) {
                    resolve(c.chan,
                            Result<SequenceId>(std::unexpected(ErrorCode::ServerShuttingDown)));
                    return;
                }
                if (c.prompt_tokens.empty() || c.max_context_len <= 0) {
                    resolve(c.chan,
                            Result<SequenceId>(std::unexpected(ErrorCode::InvalidArgument)));
                    return;
                }
                if (c.prompt_tokens.size() >
                    static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    resolve(c.chan,
                            Result<SequenceId>(std::unexpected(ErrorCode::InvalidArgument)));
                    return;
                }
                if (static_cast<int64_t>(c.max_context_len) <
                    static_cast<int64_t>(c.prompt_tokens.size())) {
                    resolve(c.chan,
                            Result<SequenceId>(std::unexpected(ErrorCode::InvalidArgument)));
                    return;
                }
                if (static_cast<int64_t>(c.max_context_len) >
                    static_cast<int64_t>(max_blocks_.load()) * kKVBlockSize) {
                    resolve(c.chan,
                            Result<SequenceId>(std::unexpected(ErrorCode::InvalidArgument)));
                    return;
                }
                for (auto tok : c.prompt_tokens) {
                    if (tok < 0) {
                        resolve(c.chan,
                                Result<SequenceId>(std::unexpected(ErrorCode::InvalidArgument)));
                        return;
                    }
                }
                if (active_sequences_.load() >= kMaxSequences) {
                    resolve(c.chan,
                            Result<SequenceId>(std::unexpected(ErrorCode::MaxSequencesReached)));
                    return;
                }

                SequenceId id = next_seq_id_++;
                SequenceState state;
                state.seq_id = id;
                state.prompt_tokens = std::move(c.prompt_tokens);
                state.max_context_len = c.max_context_len;
                sequences_[id] = std::move(state);
                active_sequences_++;
                resolve(c.chan, Result<SequenceId>{id});
            } else if constexpr (std::is_same_v<T, CmdReleaseSequence>) {
                if (!initialized_) {
                    resolve(c.chan, Result<void>{});
                    return;
                }
                auto it = sequences_.find(c.seq_id);
                if (it == sequences_.end()) {
                    resolve(c.chan, Result<void>{});
                    return;
                }
                auto r = release_sequence_state(it);
                resolve(c.chan, r);
            } else if constexpr (std::is_same_v<T, CmdAbortSequence>) {
                if (!initialized_) {
                    resolve(c.chan, Result<void>{});
                    return;
                }
                auto it = sequences_.find(c.seq_id);
                if (it == sequences_.end()) {
                    resolve(c.chan, Result<void>{});
                    return;
                }
                auto r = release_sequence_state(it);
                resolve(c.chan, r);
            }
        },
        cmd);
}

// --- Sequence state helper ---

Result<void> Worker::release_sequence_state(SeqMapIter it) {
    int blocks = static_cast<int>(it->second.block_table.size());
    if (blocks > 0) {
        auto r = kv_mgr_->release_blocks(it->second.block_table);
        if (!r) return r;
        free_blocks_ += blocks;
    }
    active_sequences_--;
    sequences_.erase(it);
    return {};
}

// --- Batch processing ---

void Worker::process_batch(PendingBatch pending) {
    if (!initialized_ || !backend_ || !kv_mgr_ || !kv_storage_) {
        resolve(pending.chan, Result<BatchResult>(std::unexpected(ErrorCode::ServerShuttingDown)));
        return;
    }

    if (pending.batch.items.empty()) {
        resolve(pending.chan, Result<BatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
        return;
    }

    // Validate all seq_ids exist before doing any work.
    for (const auto& item : pending.batch.items) {
        SequenceId seq_id = 0;
        std::visit([&](const auto& w) { seq_id = w.seq_id; }, item);
        if (sequences_.find(seq_id) == sequences_.end()) {
            resolve(pending.chan, Result<BatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
            return;
        }
    }

    if (model_) {
        BatchTranslator translator(*backend_, *kv_mgr_, kKVBlockSize);

        auto tr = translator.translate(pending.batch, sequences_);
        if (!tr) {
            resolve(pending.chan, Result<BatchResult>(std::unexpected(tr.error())));
            return;
        }

        // Count allocated blocks and update free_blocks_ immediately.
        int allocated = 0;
        for (const auto& pa : tr->per_item) {
            if (allocated > std::numeric_limits<int>::max() - pa.new_blocks.size()) {
                translator.rollback(tr->per_item);
                resolve(pending.chan,
                        Result<BatchResult>(std::unexpected(ErrorCode::InvalidArgument)));
                return;
            }
            allocated += pa.new_blocks.size();
        }
        free_blocks_ -= allocated;

        auto& phys_batch = tr->physical_batch;
        auto& per_item = tr->per_item;

        auto exec_r = ModelRunner::inference<BF16RunnerTraits>(
            *model_, phys_batch, *backend_, *kv_storage_, pending.batch.sampling);

        if (!exec_r) {
            free_blocks_ += allocated;
            translator.rollback(per_item);
            resolve(pending.chan, Result<BatchResult>(std::unexpected(exec_r.error())));
            return;
        }

        auto commit_r = translator.commit(pending.batch, sequences_, per_item);
        if (!commit_r) {
            free_blocks_ += allocated;
            translator.rollback(per_item);
            resolve(pending.chan, Result<BatchResult>(std::unexpected(commit_r.error())));
            return;
        }

        BatchResult result;
        result.batch_id = pending.batch.batch_id;
        result.items = std::move(*exec_r);
        resolve(pending.chan, Result<BatchResult>(std::move(result)));
        return;
    }

    // Dummy mode (no model loaded).
    BatchResult result = generate_dummy_result(pending.batch);
    resolve(pending.chan, Result<BatchResult>(std::move(result)));
}

BatchResult Worker::generate_dummy_result(const ScheduledBatch& batch) {
    BatchResult result;
    result.batch_id = batch.batch_id;

    for (std::size_t i = 0; i < batch.items.size(); ++i) {
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
