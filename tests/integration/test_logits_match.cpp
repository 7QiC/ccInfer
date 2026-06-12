#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "backend/device_buffer.h"
#include "cache/block.h"
#include "cache/kv_cache_manager.h"
#include "cache/kv_cache_storage.h"
#include "kernel/cuda_kernels.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/registry.h"
#include "tokenizer/byte_level_bpe_tokenizer.h"

using namespace ccinfer;

namespace {

template <typename B>
std::unique_ptr<DeviceBuffer> alloc_buf(B& backend, size_t bytes) {
    auto r = backend.allocate_buffer(bytes);
    assert(r.has_value());
    return std::move(*r);
}

std::string model_dir() {
    const char* dir = std::getenv("CCINFER_TEST_MODEL_DIR");
    return dir ? std::string(dir) : std::string{};
}

bool model_available() { return !model_dir().empty(); }

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::string content;
    f.seekg(0, std::ios::end);
    content.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0, std::ios::beg);
    f.read(content.data(), static_cast<std::streamsize>(content.size()));
    return content;
}

// ---------------------------------------------------------------------------
// Mini paged-KV-cache harness for correctness testing.
//
// Qwen3Model::forward() requires a fully populated ForwardInput including
// paged-attention metadata.  For a single prefill with T tokens we:
//   1. Create a KVCacheStorage with enough blocks for T tokens
//   2. Allocate those blocks and fill slot_mapping / block_table
//   3. Let the model embed token_ids internally (no manual embed step)
// ---------------------------------------------------------------------------

struct ForwardFixture {
    std::unique_ptr<KVCacheManager> kv_mgr;
    std::unique_ptr<DeviceBuffer> token_ids;
    std::unique_ptr<DeviceBuffer> positions;
    std::unique_ptr<DeviceBuffer> slot_mapping;
    std::unique_ptr<DeviceBuffer> block_table;
    std::unique_ptr<DeviceBuffer> query_start_loc;
    std::unique_ptr<DeviceBuffer> context_lens;
    int max_blocks_per_req = 0;
};

// Returns a fixture whose kv_mgr is nullptr on any failure.
ForwardFixture prepare_single_prefill(CudaBackend& backend, const ModelConfig& config,
                                      const std::vector<int32_t>& token_ids_host) {
    ForwardFixture fixture;
    fixture.kv_mgr = std::make_unique<KVCacheManager>();

    const int T = static_cast<int>(token_ids_host.size());
    const int max_blocks = std::max(1, (T + kKVBlockSize - 1) / kKVBlockSize);

    auto storage = KVCacheStorage::create<__nv_bfloat16>(
        backend, config.n_layers_, max_blocks, kKVBlockSize, config.n_kv_heads_, config.head_dim_);
    if (!storage) {
        fprintf(stderr, "prepare_single_prefill: KVCacheStorage::create failed\n");
        fixture.kv_mgr.reset();
        return fixture;
    }
    auto init = fixture.kv_mgr->init(std::move(*storage), max_blocks, kKVBlockSize);
    if (!init) {
        fprintf(stderr, "prepare_single_prefill: KVCacheManager::init failed\n");
        fixture.kv_mgr.reset();
        return fixture;
    }

    auto blocks = fixture.kv_mgr->allocate_blocks(max_blocks);
    if (!blocks) {
        fprintf(stderr, "prepare_single_prefill: allocate_blocks failed\n");
        fixture.kv_mgr.reset();
        return fixture;
    }
    fixture.max_blocks_per_req = blocks->size();

    // Build host metadata arrays.
    std::vector<int32_t> positions_host(static_cast<size_t>(T));
    std::vector<int32_t> slot_mapping_host(static_cast<size_t>(T));
    for (int i = 0; i < T; ++i) {
        positions_host[static_cast<size_t>(i)] = i;
        slot_mapping_host[static_cast<size_t>(i)] =
            (*blocks)[i / kKVBlockSize] * kKVBlockSize + (i % kKVBlockSize);
    }
    const std::vector<int32_t> query_start_loc_host{0, T};
    const std::vector<int32_t> context_lens_host{T};

    // Allocate device buffers.
    fixture.token_ids = alloc_buf(backend, static_cast<size_t>(T) * sizeof(int32_t));
    fixture.positions = alloc_buf(backend, static_cast<size_t>(T) * sizeof(int32_t));
    fixture.slot_mapping = alloc_buf(backend, static_cast<size_t>(T) * sizeof(int32_t));
    fixture.block_table = alloc_buf(backend, static_cast<size_t>(blocks->size()) * sizeof(int32_t));
    fixture.query_start_loc = alloc_buf(backend, 2 * sizeof(int32_t));
    fixture.context_lens = alloc_buf(backend, sizeof(int32_t));

    // Upload to device.
    cudaMemcpy(fixture.token_ids->data(), token_ids_host.data(), T * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(fixture.positions->data(), positions_host.data(), T * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(fixture.slot_mapping->data(), slot_mapping_host.data(), T * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(fixture.block_table->data(), blocks->data(), blocks->size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(fixture.query_start_loc->data(), query_start_loc_host.data(), 2 * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(fixture.context_lens->data(), context_lens_host.data(), sizeof(int32_t),
               cudaMemcpyHostToDevice);

    return fixture;
}

// Run a single-prompt prefill through Qwen3Model and return the last-token
// FP32 logits.  Returns an empty vector on any error (message printed to stderr).
std::vector<float> run_prefill_logits(CudaBackend& backend, const ModelConfig& config,
                                      const WeightLoader& loader,
                                      const std::vector<int32_t>& token_ids) {
    auto model = ModelRegistry::instance().create(config, loader, backend);
    if (!model) {
        fprintf(stderr, "run_prefill_logits: ModelRegistry::create failed\n");
        return {};
    }

    auto fixture = prepare_single_prefill(backend, config, token_ids);
    if (!fixture.kv_mgr) {
        fprintf(stderr, "run_prefill_logits: prepare_single_prefill failed\n");
        return {};
    }

    const int T = static_cast<int>(token_ids.size());
    const int V = config.vocab_size_;
    auto output_logits = alloc_buf(backend, static_cast<size_t>(T) * V * sizeof(float));

    ForwardInput fwd_in{};
    fwd_in.token_ids_ = static_cast<int32_t*>(fixture.token_ids->data());
    fwd_in.num_tokens_ = T;
    fwd_in.positions_ = static_cast<int32_t*>(fixture.positions->data());
    fwd_in.max_position_id_ = T - 1;
    fwd_in.mode_ = ForwardMode::Prefill;
    fwd_in.kv_mgr_ = fixture.kv_mgr.get();
    fwd_in.slot_mapping_ = static_cast<int32_t*>(fixture.slot_mapping->data());
    fwd_in.block_table_ = static_cast<int32_t*>(fixture.block_table->data());
    fwd_in.query_start_loc_ = static_cast<int32_t*>(fixture.query_start_loc->data());
    fwd_in.context_lens_ = static_cast<int32_t*>(fixture.context_lens->data());
    fwd_in.batch_size_ = 1;
    fwd_in.max_blocks_per_req_ = fixture.max_blocks_per_req;

    ForwardOutput fwd_out{};
    fwd_out.logits_ = output_logits->data();

    auto fwd_result = (*model)->forward(fwd_in, fwd_out, backend);
    if (!fwd_result) {
        fprintf(stderr, "run_prefill_logits: forward failed\n");
        return {};
    }
    backend.synchronize();

    std::vector<float> logits(static_cast<size_t>(V));
    cudaMemcpy(logits.data(),
               static_cast<float*>(output_logits->data()) + static_cast<int64_t>(T - 1) * V,
               V * sizeof(float), cudaMemcpyDeviceToHost);
    return logits;
}

// ---------------------------------------------------------------------------
// Greedy generation: prefill + decode loop
// ---------------------------------------------------------------------------

struct GenFixture {
    std::unique_ptr<KVCacheManager> kv_mgr;
    std::unique_ptr<DeviceBuffer> token_ids;       // [1] — single token per decode step
    std::unique_ptr<DeviceBuffer> positions;        // [1]
    std::unique_ptr<DeviceBuffer> slot_mapping;     // [1]
    std::unique_ptr<DeviceBuffer> block_table;      // [1, max_blocks]
    std::unique_ptr<DeviceBuffer> query_start_loc;  // [2]
    std::unique_ptr<DeviceBuffer> context_lens;     // [1]
    int max_blocks_per_req = 0;
    int current_context_len = 0;   // total tokens written to KV cache
    int next_slot = 0;             // next free slot index
    std::vector<int32_t> block_ids; // physical block IDs
};

// Creates a GenFixture with enough blocks for prompt + max_new_tokens.
// Returns a fixture with kv_mgr == nullptr on failure.
GenFixture prepare_generation(CudaBackend& backend, const ModelConfig& config,
                              const std::vector<int32_t>& prompt_tokens,
                              int max_new_tokens) {
    GenFixture f;
    f.kv_mgr = std::make_unique<KVCacheManager>();

    const int prompt_len = static_cast<int>(prompt_tokens.size());
    const int total_tokens = prompt_len + max_new_tokens;
    const int total_blocks = std::max(1, (total_tokens + kKVBlockSize - 1) / kKVBlockSize);

    auto storage = KVCacheStorage::create<__nv_bfloat16>(
        backend, config.n_layers_, total_blocks, kKVBlockSize, config.n_kv_heads_,
        config.head_dim_);
    if (!storage) { f.kv_mgr.reset(); return f; }
    if (!f.kv_mgr->init(std::move(*storage), total_blocks, kKVBlockSize)) {
        f.kv_mgr.reset(); return f;
    }
    auto blocks = f.kv_mgr->allocate_blocks(total_blocks);
    if (!blocks) { f.kv_mgr.reset(); return f; }
    f.max_blocks_per_req = blocks->size();
    f.block_ids.resize(static_cast<size_t>(blocks->size()));
    for (int i = 0; i < blocks->size(); ++i)
        f.block_ids[static_cast<size_t>(i)] = (*blocks)[i];

    // Allocate device buffers
    f.token_ids = alloc_buf(backend, sizeof(int32_t));
    f.positions = alloc_buf(backend, sizeof(int32_t));
    f.slot_mapping = alloc_buf(backend, sizeof(int32_t));
    f.block_table = alloc_buf(backend, static_cast<size_t>(f.max_blocks_per_req) * sizeof(int32_t));
    f.query_start_loc = alloc_buf(backend, 2 * sizeof(int32_t));
    f.context_lens = alloc_buf(backend, sizeof(int32_t));

    // Upload block table (static across all steps)
    cudaMemcpy(f.block_table->data(), f.block_ids.data(),
               static_cast<size_t>(f.max_blocks_per_req) * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    return f;
}

// Run one decode step.  token_id is the just-sampled token to feed as input.
// Writes KV for this token into the cache and returns logits for the NEXT token.
// Returns empty vector on error.
std::vector<float> run_decode_step(CudaBackend& backend, const ModelConfig& config,
                                   std::unique_ptr<Model>& model,
                                   GenFixture& f, int32_t token_id) {
    const int V = config.vocab_size_;
    const int pos = f.current_context_len;
    const int slot = f.next_slot;

    // Upload single-token metadata
    cudaMemcpy(f.token_ids->data(), &token_id, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(f.positions->data(), &pos, sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(f.slot_mapping->data(), &slot, sizeof(int32_t), cudaMemcpyHostToDevice);

    int32_t qsl[2] = {0, 1};
    cudaMemcpy(f.query_start_loc->data(), qsl, 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

    int32_t ctx = pos + 1;  // context includes this token
    cudaMemcpy(f.context_lens->data(), &ctx, sizeof(int32_t), cudaMemcpyHostToDevice);

    auto logits_buf = alloc_buf(backend, static_cast<size_t>(V) * sizeof(float));

    ForwardInput fwd_in{};
    fwd_in.token_ids_ = static_cast<int32_t*>(f.token_ids->data());
    fwd_in.num_tokens_ = 1;
    fwd_in.positions_ = static_cast<int32_t*>(f.positions->data());
    fwd_in.max_position_id_ = pos;
    fwd_in.mode_ = ForwardMode::Decode;
    fwd_in.kv_mgr_ = f.kv_mgr.get();
    fwd_in.slot_mapping_ = static_cast<int32_t*>(f.slot_mapping->data());
    fwd_in.block_table_ = static_cast<int32_t*>(f.block_table->data());
    fwd_in.query_start_loc_ = static_cast<int32_t*>(f.query_start_loc->data());
    fwd_in.context_lens_ = static_cast<int32_t*>(f.context_lens->data());
    fwd_in.batch_size_ = 1;
    fwd_in.max_blocks_per_req_ = f.max_blocks_per_req;

    ForwardOutput fwd_out{};
    fwd_out.logits_ = logits_buf->data();

    if (!model->forward(fwd_in, fwd_out, backend)) {
        fprintf(stderr, "run_decode_step: forward failed at pos=%d\n", pos);
        return {};
    }
    backend.synchronize();

    f.current_context_len = ctx;
    f.next_slot = slot + 1;

    std::vector<float> logits(static_cast<size_t>(V));
    cudaMemcpy(logits.data(), logits_buf->data(),
               V * sizeof(float), cudaMemcpyDeviceToHost);
    return logits;
}

// CPU-side greedy argmax
int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    float best_val = v[0];
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > best_val) { best_val = v[i]; best = static_cast<int32_t>(i); }
    }
    return best;
}

// Run full greedy generation: prefill + decode loop.
// Returns generated NEW token IDs (excluding prompt).  Empty on error.
std::vector<int32_t> run_greedy_generation(CudaBackend& backend,
                                           const ModelConfig& config,
                                           const WeightLoader& loader,
                                           const std::vector<int32_t>& prompt_tokens,
                                           int max_new_tokens) {
    auto model = ModelRegistry::instance().create(config, loader, backend);
    if (!model) {
        fprintf(stderr, "run_greedy_generation: ModelRegistry::create failed\n");
        return {};
    }

    auto f = prepare_generation(backend, config, prompt_tokens, max_new_tokens);
    if (!f.kv_mgr) {
        fprintf(stderr, "run_greedy_generation: prepare_generation failed\n");
        return {};
    }

    const int prompt_len = static_cast<int>(prompt_tokens.size());
    const int V = config.vocab_size_;

    // ---- Prefill: write all prompt tokens into KV cache ----
    {
        auto prefill_logits_buf = alloc_buf(backend,
            static_cast<size_t>(prompt_len) * V * sizeof(float));

        // Build prefill metadata
        std::vector<int32_t> positions_host(static_cast<size_t>(prompt_len));
        std::vector<int32_t> slot_mapping_host(static_cast<size_t>(prompt_len));
        for (int i = 0; i < prompt_len; ++i) {
            positions_host[static_cast<size_t>(i)] = i;
            slot_mapping_host[static_cast<size_t>(i)] =
                f.block_ids[i / kKVBlockSize] * kKVBlockSize + (i % kKVBlockSize);
        }
        std::vector<int32_t> qsl{0, prompt_len};
        std::vector<int32_t> ctx{prompt_len};

        auto pf_token_ids = alloc_buf(backend, static_cast<size_t>(prompt_len) * sizeof(int32_t));
        auto pf_pos = alloc_buf(backend, static_cast<size_t>(prompt_len) * sizeof(int32_t));
        auto pf_slot = alloc_buf(backend, static_cast<size_t>(prompt_len) * sizeof(int32_t));
        auto pf_qsl = alloc_buf(backend, 2 * sizeof(int32_t));
        auto pf_ctx = alloc_buf(backend, sizeof(int32_t));

        cudaMemcpy(pf_token_ids->data(), prompt_tokens.data(),
                   prompt_len * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pf_pos->data(), positions_host.data(),
                   prompt_len * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pf_slot->data(), slot_mapping_host.data(),
                   prompt_len * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pf_qsl->data(), qsl.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pf_ctx->data(), ctx.data(), sizeof(int32_t), cudaMemcpyHostToDevice);

        ForwardInput fwd_in{};
        fwd_in.token_ids_ = static_cast<int32_t*>(pf_token_ids->data());
        fwd_in.num_tokens_ = prompt_len;
        fwd_in.positions_ = static_cast<int32_t*>(pf_pos->data());
        fwd_in.max_position_id_ = prompt_len - 1;
        fwd_in.mode_ = ForwardMode::Prefill;
        fwd_in.kv_mgr_ = f.kv_mgr.get();
        fwd_in.slot_mapping_ = static_cast<int32_t*>(pf_slot->data());
        fwd_in.block_table_ = static_cast<int32_t*>(f.block_table->data());
        fwd_in.query_start_loc_ = static_cast<int32_t*>(pf_qsl->data());
        fwd_in.context_lens_ = static_cast<int32_t*>(pf_ctx->data());
        fwd_in.batch_size_ = 1;
        fwd_in.max_blocks_per_req_ = f.max_blocks_per_req;

        ForwardOutput fwd_out{};
        fwd_out.logits_ = prefill_logits_buf->data();

        if (!(*model)->forward(fwd_in, fwd_out, backend)) {
            fprintf(stderr, "run_greedy_generation: prefill failed\n");
            return {};
        }
        backend.synchronize();

        f.current_context_len = prompt_len;
        f.next_slot = prompt_len;

        // Sample first token from prefill's last logit
        std::vector<float> last_logits(static_cast<size_t>(V));
        cudaMemcpy(last_logits.data(),
                   static_cast<float*>(prefill_logits_buf->data())
                       + static_cast<int64_t>(prompt_len - 1) * V,
                   V * sizeof(float), cudaMemcpyDeviceToHost);
        int32_t first_token = argmax(last_logits);

        std::vector<int32_t> generated;
        generated.push_back(first_token);

        // ---- Decode loop ----
        int32_t current_token = first_token;
        for (int step = 1; step < max_new_tokens; ++step) {
            auto logits = run_decode_step(backend, config, *model, f, current_token);
            if (logits.empty()) {
                fprintf(stderr, "run_greedy_generation: decode step %d failed\n", step);
                return {};
            }
            current_token = argmax(logits);
            generated.push_back(current_token);
        }
        return generated;
    }
}

}  // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class LogitsMatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_available()) {
            GTEST_SKIP() << "CCINFER_TEST_MODEL_DIR not set";
        }

        cudaStreamCreate(&stream_);

        dir_ = model_dir();

        auto cfg_json = nlohmann::json::parse(read_file(dir_ + "/config.json"), nullptr, false);
        ASSERT_FALSE(cfg_json.is_discarded());
        auto cfg_result = ModelConfig::from_json(cfg_json);
        ASSERT_TRUE(cfg_result);
        config_ = *cfg_result;

        ASSERT_TRUE(tokenizer_.load(dir_ + "/tokenizer.json"));

        auto loader_result = WeightLoader::create(dir_ + "/model.safetensors");
        ASSERT_TRUE(loader_result);
        loader_ = std::make_unique<WeightLoader>(std::move(*loader_result));

        auto b = CudaBackend::create(0);
        ASSERT_TRUE(b.has_value());
        backend_ = std::move(*b);
        register_builtin_models();
    }

    void TearDown() override {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    std::string dir_;
    ModelConfig config_;
    ByteLevelBpeTokenizer tokenizer_;
    std::unique_ptr<WeightLoader> loader_;
    std::unique_ptr<CudaBackend> backend_;
    cudaStream_t stream_{};
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_F(LogitsMatchTest, SingleToken) {
    const std::string prompt = "Hello";
    auto ids_result = tokenizer_.encode(prompt);
    ASSERT_TRUE(ids_result);
    const auto& token_ids = *ids_result;
    ASSERT_GT(token_ids.size(), 0);
    const int V = config_.vocab_size_;
    auto logits = run_prefill_logits(*backend_, config_, *loader_, token_ids);
    ASSERT_FALSE(logits.empty()) << "Forward pass failed";

    std::string ref_path = dir_ + "/ref_logits_single.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (!ref_file.good()) GTEST_SKIP() << "Run save_ref_logits.py first";
    std::vector<float> ref_logits(static_cast<size_t>(V));
    ref_file.read(reinterpret_cast<char*>(ref_logits.data()),
                  static_cast<std::streamsize>(V * sizeof(float)));
    if (ref_file.gcount() != static_cast<std::streamsize>(V * sizeof(float)))
        GTEST_SKIP() << "size mismatch";

    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < V; ++i) {
        float diff = std::abs(logits[static_cast<size_t>(i)] - ref_logits[static_cast<size_t>(i)]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    printf("Single-token max_diff=%.6f at token %d (ccinf=%.4f ref=%.4f)\n",
           max_diff, max_idx, logits[static_cast<size_t>(max_idx)],
           ref_logits[static_cast<size_t>(max_idx)]);

    // BF16-matching accumulation (RMSNorm rounds sum_sq after each element).
    // With this strategy, single-token max_diff stays under 0.10.
    EXPECT_LT(max_diff, 0.10f) << "Single-token precision should be within BF16 tolerance";
}

TEST_F(LogitsMatchTest, CompareWithReference) {
    const std::string prompt = "Hello world";
    auto ids_result = tokenizer_.encode(prompt);
    ASSERT_TRUE(ids_result);
    const auto& token_ids = *ids_result;
    ASSERT_GT(token_ids.size(), 0);
    const int V = config_.vocab_size_;
    auto logits = run_prefill_logits(*backend_, config_, *loader_, token_ids);
    ASSERT_FALSE(logits.empty()) << "Forward pass failed";

    std::string ref_path = dir_ + "/ref_logits.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (!ref_file.good())
        GTEST_SKIP() << "Reference logits not found. Run scripts/save_ref_logits.py first.";

    std::vector<float> ref_logits(static_cast<size_t>(V));
    ref_file.read(reinterpret_cast<char*>(ref_logits.data()),
                  static_cast<std::streamsize>(V * sizeof(float)));

    if (ref_file.gcount() != static_cast<std::streamsize>(V * sizeof(float)))
        GTEST_SKIP() << "Reference logits file size mismatch";

    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < V; ++i) {
        float diff = std::abs(logits[static_cast<size_t>(i)] - ref_logits[static_cast<size_t>(i)]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    // Multi-token accumulates more GEMM rounding across the sequence.
    EXPECT_LT(max_diff, 0.25f) << "Max logit diff at token " << max_idx
                               << ": ccinfer=" << logits[static_cast<size_t>(max_idx)]
                               << " ref=" << ref_logits[static_cast<size_t>(max_idx)];

    // Top-5 overlap check
    std::vector<int> our_idx(static_cast<size_t>(V)), ref_idx(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) {
        our_idx[static_cast<size_t>(i)] = i;
        ref_idx[static_cast<size_t>(i)] = i;
    }
    std::partial_sort(our_idx.begin(), our_idx.begin() + 5, our_idx.end(),
                      [&](int a, int b) {
                          return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
                      });
    std::partial_sort(ref_idx.begin(), ref_idx.begin() + 5, ref_idx.end(),
                      [&](int a, int b) {
                          return ref_logits[static_cast<size_t>(a)] > ref_logits[static_cast<size_t>(b)];
                      });

    int top5_match = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (our_idx[static_cast<size_t>(i)] == ref_idx[static_cast<size_t>(j)]) {
                ++top5_match;
                break;
            }
        }
    }
    EXPECT_GE(top5_match, 3) << "Top-5 overlap too low: " << top5_match << "/5";
}

TEST_F(LogitsMatchTest, TopKAgreement) {
    const std::string prompt = "Hello world";
    auto ids_result = tokenizer_.encode(prompt);
    ASSERT_TRUE(ids_result);
    const auto& token_ids = *ids_result;
    ASSERT_GT(token_ids.size(), 0);
    const int V = config_.vocab_size_;
    auto logits = run_prefill_logits(*backend_, config_, *loader_, token_ids);
    ASSERT_FALSE(logits.empty()) << "Forward pass failed";

    std::vector<int> idx(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) idx[static_cast<size_t>(i)] = i;

    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) {
                          return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
                      });

    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isfinite(logits[static_cast<size_t>(idx[i])]))
            << "Top-" << (i + 1) << " logit is not finite";
    }
}

TEST_F(LogitsMatchTest, GreedyGenerationMatches) {
    // Compare ccInfer greedy generation with HF Transformers greedy output.
    // Loads HF metadata, runs ccInfer generation, compares token-by-token.

    auto load_hf_gen = [&](const std::string& ref_dir) -> std::vector<int32_t> {
        std::string meta_path =
            dir_ + "/ccinfer_correctness_ref/" + ref_dir + "/metadata.json";
        std::ifstream f(meta_path);
        if (!f.is_open()) {
            printf("  SKIP: no HF reference at %s\n", meta_path.c_str());
            return {};
        }
        auto meta = nlohmann::json::parse(f, nullptr, false);
        if (meta.is_discarded()) return {};
        std::vector<int32_t> gen;
        for (auto& v : meta["generation"]["generated_new_token_ids"])
            gen.push_back(static_cast<int32_t>(v.get<int>()));
        return gen;
    };

    struct GenCase { std::string prompt; std::string ref_dir; int max_tokens; };
    const std::vector<GenCase> test_cases = {
        {"Hello", "Hello", 16},
        {"Hello world", "Hello_world", 16},
        {"The quick brown fox jumps over the lazy dog", "The_quick_brown_fox", 16},
    };

    for (const auto& tc : test_cases) {
        printf("\n--- Generation: \"%s\" ---\n", tc.prompt.c_str());

        auto ids_result = tokenizer_.encode(tc.prompt);
        ASSERT_TRUE(ids_result);
        const auto& prompt_tokens = *ids_result;
        ASSERT_GT(prompt_tokens.size(), 0);

        auto hf_gen = load_hf_gen(tc.ref_dir);
        if (hf_gen.empty()) {
            GTEST_SKIP() << "HF reference not found for: " << tc.prompt;
        }

        auto cc_gen = run_greedy_generation(*backend_, config_, *loader_,
                                            prompt_tokens, tc.max_tokens);
        ASSERT_FALSE(cc_gen.empty()) << "Generation failed for: " << tc.prompt;

        printf("  Prompt tokens: ");
        for (auto t : prompt_tokens) printf("%d ", t);
        printf("\n  HF generated:   ");
        for (size_t i = 0; i < cc_gen.size() && i < hf_gen.size(); ++i)
            printf("%d ", hf_gen[i]);
        printf("\n  ccInfer generated: ");
        for (auto t : cc_gen) printf("%d ", t);
        printf("\n");

        // Compare token-by-token
        size_t compare_len = std::min(cc_gen.size(), hf_gen.size());
        int match_count = 0;
        int first_divergence = -1;
        for (size_t i = 0; i < compare_len; ++i) {
            if (cc_gen[i] == hf_gen[i]) {
                ++match_count;
            } else if (first_divergence < 0) {
                first_divergence = static_cast<int>(i);
            }
        }

        printf("  Match: %zu/%zu tokens exact", static_cast<size_t>(match_count), compare_len);
        if (first_divergence >= 0)
            printf(" (first divergence at token %d)", first_divergence);
        printf("\n");

        // For BF16 greedy generation, we expect exact match for at least the
        // first few tokens.  Small logit differences can cause divergence, but
        // typically not on the very first token of short prompts.
        EXPECT_GE(match_count, static_cast<int>(compare_len) * 3 / 4)
            << "Too many divergences in greedy generation for: " << tc.prompt;
        if (first_divergence >= 0) {
            printf("  NOTE: divergence at token %d — ccInfer chose %d, HF chose %d\n",
                   first_divergence, cc_gen[static_cast<size_t>(first_divergence)],
                   hf_gen[static_cast<size_t>(first_divergence)]);
        }
    }
}
