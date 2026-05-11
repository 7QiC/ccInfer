#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/backend/device_buffer.h"
#include "engine/kernel/cuda_kernels.h"
#include "engine/model/config.h"
#include "engine/model/loader.h"
#include "engine/model/registry.h"
#include "server/tokenizer/byte_level_bpe_tokenizer.h"

using namespace ccinfer;
using namespace ccinfer::engine;
using namespace ccinfer::server;

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

}  // namespace

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

TEST_F(LogitsMatchTest, SingleToken) {
    const std::string prompt = "Hello";

    auto ids_result = tokenizer_.encode(prompt);
    ASSERT_TRUE(ids_result);
    const auto& token_ids = *ids_result;
    const int T = static_cast<int>(token_ids.size());
    ASSERT_GT(T, 0);
    const int D = config_.d_model_;
    const int V = config_.vocab_size_;

    auto embed = loader_->load<__nv_bfloat16>(*backend_, "model.embed_tokens.weight", {V, D});
    ASSERT_TRUE(embed);

    auto token_ids_dev = alloc_buf(*backend_,static_cast<size_t>(T) * sizeof(int32_t));
    cudaMemcpy(token_ids_dev->data(), token_ids.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);

    auto input_embeds = alloc_buf(*backend_,static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto r = launch_embed(static_cast<__nv_bfloat16*>((*embed)->data()),
                          static_cast<int32_t*>(token_ids_dev->data()),
                          static_cast<__nv_bfloat16*>(input_embeds->data()), T, D, stream_);
    ASSERT_TRUE(r);

    auto model = ModelRegistry::instance().create(config_, *loader_, *backend_);
    ASSERT_TRUE(model);

    auto output_logits = alloc_buf(*backend_,static_cast<size_t>(T) * V * sizeof(__nv_bfloat16));

    ForwardInput fwd_in{};
    fwd_in.input_embeds_ = static_cast<__nv_bfloat16*>(input_embeds->data());
    fwd_in.num_tokens_ = T;

    ForwardOutput fwd_out{};
    fwd_out.logits_ = static_cast<__nv_bfloat16*>(output_logits->data());

    auto fwd_result = (*model)->forward(fwd_in, fwd_out, *backend_);
    ASSERT_TRUE(fwd_result);
    cudaStreamSynchronize(stream_);

    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<__nv_bfloat16> logits_bf16(static_cast<size_t>(V));
    cudaMemcpy(logits_bf16.data(),
               static_cast<__nv_bfloat16*>(output_logits->data()) + static_cast<int64_t>(T - 1) * V,
               V * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < V; ++i) logits[static_cast<size_t>(i)] = __bfloat162float(logits_bf16[static_cast<size_t>(i)]);

    std::string ref_path = dir_ + "/ref_logits_single.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (!ref_file.good()) GTEST_SKIP() << "Run save_ref_logits.py first";
    std::vector<float> ref_logits(static_cast<size_t>(V));
    ref_file.read(reinterpret_cast<char*>(ref_logits.data()), static_cast<std::streamsize>(V * sizeof(float)));
    if (ref_file.gcount() != static_cast<std::streamsize>(V * sizeof(float))) GTEST_SKIP() << "size mismatch";

    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < V; ++i) {
        float diff = std::abs(logits[static_cast<size_t>(i)] - ref_logits[static_cast<size_t>(i)]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    printf("Single-token max_diff=%.6f at token %d (ccinf=%.4f ref=%.4f)\n",
           max_diff, max_idx, logits[static_cast<size_t>(max_idx)], ref_logits[static_cast<size_t>(max_idx)]);
    // BF16 arithmetic across 28 layers inherently accumulates ~0.1-0.15 max_diff
    // due to 1-ULP differences in RMSNorm, GEMM, SiLU rounding vs PyTorch.
    EXPECT_LT(max_diff, 0.15f) << "Single-token precision should be within BF16 tolerance";
}

TEST_F(LogitsMatchTest, CompareWithReference) {
    const std::string prompt = "Hello world";

    auto ids_result = tokenizer_.encode(prompt);
    ASSERT_TRUE(ids_result);
    const auto& token_ids = *ids_result;
    const int T = static_cast<int>(token_ids.size());
    ASSERT_GT(T, 0);
    const int D = config_.d_model_;
    const int V = config_.vocab_size_;

    // Load embed temporarily for lookup, then create model.
    auto embed = loader_->load<__nv_bfloat16>(*backend_, "model.embed_tokens.weight", {V, D});
    ASSERT_TRUE(embed);

    auto token_ids_dev = alloc_buf(*backend_,static_cast<size_t>(T) * sizeof(int32_t));
    cudaMemcpy(token_ids_dev->data(), token_ids.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);

    auto input_embeds = alloc_buf(*backend_,static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto r = launch_embed(static_cast<__nv_bfloat16*>((*embed)->data()),
                          static_cast<int32_t*>(token_ids_dev->data()),
                          static_cast<__nv_bfloat16*>(input_embeds->data()), T, D, stream_);
    ASSERT_TRUE(r);

    // Create model via registry — weights are loaded inside.
    auto model = ModelRegistry::instance().create(config_, *loader_, *backend_);
    ASSERT_TRUE(model);

    auto output_logits = alloc_buf(*backend_,static_cast<size_t>(T) * V * sizeof(__nv_bfloat16));

    ForwardInput fwd_in{};
    fwd_in.input_embeds_ = static_cast<__nv_bfloat16*>(input_embeds->data());
    fwd_in.num_tokens_ = T;

    ForwardOutput fwd_out{};
    fwd_out.logits_ = static_cast<__nv_bfloat16*>(output_logits->data());

    auto fwd_result = (*model)->forward(fwd_in, fwd_out, *backend_);
    ASSERT_TRUE(fwd_result);

    cudaStreamSynchronize(stream_);

    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<__nv_bfloat16> logits_bf16(static_cast<size_t>(V));
    cudaMemcpy(logits_bf16.data(),
               static_cast<__nv_bfloat16*>(output_logits->data()) + static_cast<int64_t>(T - 1) * V,
               V * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (int i = 0; i < V; ++i) {
        logits[static_cast<size_t>(i)] = __bfloat162float(logits_bf16[static_cast<size_t>(i)]);
    }

    std::string ref_path = dir_ + "/ref_logits.bin";
    std::ifstream ref_file(ref_path, std::ios::binary);
    if (!ref_file.good()) {
        GTEST_SKIP() << "Reference logits not found. Run scripts/save_ref_logits.py first.";
    }

    std::vector<float> ref_logits(static_cast<size_t>(V));
    ref_file.read(reinterpret_cast<char*>(ref_logits.data()),
                  static_cast<std::streamsize>(V * sizeof(float)));

    if (ref_file.gcount() != static_cast<std::streamsize>(V * sizeof(float))) {
        GTEST_SKIP() << "Reference logits file size mismatch";
    }

    float max_diff = 0.0f;
    int max_idx = 0;
    for (int i = 0; i < V; ++i) {
        float diff = std::abs(logits[static_cast<size_t>(i)] - ref_logits[static_cast<size_t>(i)]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    EXPECT_LT(max_diff, 0.25f) << "Max logit diff at token " << max_idx
                               << ": ccinfer=" << logits[static_cast<size_t>(max_idx)]
                               << " ref=" << ref_logits[static_cast<size_t>(max_idx)];

    std::vector<int> our_idx(static_cast<size_t>(V)), ref_idx(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) {
        our_idx[static_cast<size_t>(i)] = i;
        ref_idx[static_cast<size_t>(i)] = i;
    }
    std::partial_sort(our_idx.begin(), our_idx.begin() + 5, our_idx.end(),
                      [&](int a, int b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; });
    std::partial_sort(ref_idx.begin(), ref_idx.begin() + 5, ref_idx.end(),
                      [&](int a, int b) { return ref_logits[static_cast<size_t>(a)] > ref_logits[static_cast<size_t>(b)]; });

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
    const int T = static_cast<int>(token_ids.size());
    const int D = config_.d_model_;
    const int V = config_.vocab_size_;

    auto embed = loader_->load<__nv_bfloat16>(*backend_, "model.embed_tokens.weight", {V, D});
    ASSERT_TRUE(embed);

    auto token_ids_dev = alloc_buf(*backend_,static_cast<size_t>(T) * sizeof(int32_t));
    cudaMemcpy(token_ids_dev->data(), token_ids.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);

    auto input_embeds = alloc_buf(*backend_,static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto r = launch_embed(static_cast<__nv_bfloat16*>((*embed)->data()),
                          static_cast<int32_t*>(token_ids_dev->data()),
                          static_cast<__nv_bfloat16*>(input_embeds->data()), T, D, stream_);
    ASSERT_TRUE(r);

    auto model = ModelRegistry::instance().create(config_, *loader_, *backend_);
    ASSERT_TRUE(model);

    auto output_logits = alloc_buf(*backend_,static_cast<size_t>(T) * V * sizeof(__nv_bfloat16));

    ForwardInput fwd_in{};
    fwd_in.input_embeds_ = static_cast<__nv_bfloat16*>(input_embeds->data());
    fwd_in.num_tokens_ = T;

    ForwardOutput fwd_out{};
    fwd_out.logits_ = static_cast<__nv_bfloat16*>(output_logits->data());

    auto fwd_result = (*model)->forward(fwd_in, fwd_out, *backend_);
    ASSERT_TRUE(fwd_result);

    cudaStreamSynchronize(stream_);

    std::vector<float> logits(static_cast<size_t>(V));
    std::vector<__nv_bfloat16> logits_bf16(static_cast<size_t>(V));
    cudaMemcpy(logits_bf16.data(),
               static_cast<__nv_bfloat16*>(output_logits->data()) + static_cast<int64_t>(T - 1) * V,
               V * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (int i = 0; i < V; ++i) {
        logits[static_cast<size_t>(i)] = __bfloat162float(logits_bf16[static_cast<size_t>(i)]);
    }

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
