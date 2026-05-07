#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "engine/backend/backend.h"
#include "engine/backend/cuda/cuda_backend.h"
#include "engine/backend/device_buffer.h"
#include "engine/kernel/cuda_kernels.h"
#include "engine/model/config.h"
#include "engine/model/loader.h"
#include "engine/model/registry.h"
#include "engine/model/rope/rope_cache.h"
#include "engine/tokenizer/byte_level_bpe_tokenizer.h"

using namespace ccinfer;
using namespace ccinfer::engine;

namespace {

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

void save_bin(const std::string& path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

}  // namespace

class LayerMatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_available()) GTEST_SKIP() << "CCINFER_TEST_MODEL_DIR not set";
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
    }

    void TearDown() override {
        if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
    }

    std::string dir_;
    ModelConfig config_;
    ByteLevelBpeTokenizer tokenizer_;
    std::unique_ptr<WeightLoader> loader_;
    CudaBackend backend_;
    cudaStream_t stream_{};
};

// Dump ccInfer hidden states after each layer + final norm to raw FP32 files.
TEST_F(LayerMatchTest, DumpLayerOutputs) {
    const std::string prompt = "Hello";
    auto ids = tokenizer_.encode(prompt);
    ASSERT_TRUE(ids);
    const auto& tokens = *ids;
    const int T = static_cast<int>(tokens.size());
    const int D = config_.d_model_;
    const int V = config_.vocab_size_;
    const int nq = config_.n_q_heads_;
    const int nkv = config_.n_kv_heads_;
    const int hd = config_.head_dim_;
    const int d_ff = config_.d_ff_;
    const int n_layers = config_.n_layers_;
    const float eps = config_.rms_norm_eps_;
    const int qkv_dim = (nq + 2 * nkv) * hd;
    const int attn_dim = nq * hd;

    auto embed = loader_->load<__nv_bfloat16>(backend_, "model.embed_tokens.weight", {V, D});
    ASSERT_TRUE(embed);
    auto rms_final = loader_->load<__nv_bfloat16>(backend_, "model.norm.weight", {D});
    ASSERT_TRUE(rms_final);
    auto lm_head = loader_->load<__nv_bfloat16>(backend_, "lm_head.weight", {V, D});
    ASSERT_TRUE(lm_head);

    // Build rope cache
    auto rc = RopeCache::create(config_.max_seq_len_, hd, config_.rope_theta_, backend_);
    ASSERT_TRUE(rc);
    auto& rope_cache = *rc;

    // Position buffer
    auto pos_buf = backend_.allocate_buffer(static_cast<size_t>(T) * sizeof(int32_t));
    {
        std::vector<int32_t> pos_host(static_cast<size_t>(T));
        for (int i = 0; i < T; ++i) pos_host[static_cast<size_t>(i)] = i;
        cudaMemcpy(pos_buf->data(), pos_host.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    // Token ids → input embeds
    auto token_ids_dev = backend_.allocate_buffer(static_cast<size_t>(T) * sizeof(int32_t));
    cudaMemcpy(token_ids_dev->data(), tokens.data(), T * sizeof(int32_t), cudaMemcpyHostToDevice);

    auto input_embeds = backend_.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    ASSERT_TRUE(launch_embed(static_cast<__nv_bfloat16*>((*embed)->data()), static_cast<int32_t*>(token_ids_dev->data()), static_cast<__nv_bfloat16*>(input_embeds->data()), T, D, stream_));

    // Work buffers
    auto hidden_a = backend_.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto hidden_b = backend_.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto normed = backend_.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto qkv_out = backend_.allocate_buffer(static_cast<size_t>(T) * qkv_dim * sizeof(__nv_bfloat16));
    auto q_buf = backend_.allocate_buffer(static_cast<size_t>(T) * nq * hd * sizeof(__nv_bfloat16));
    auto k_buf = backend_.allocate_buffer(static_cast<size_t>(T) * nkv * hd * sizeof(__nv_bfloat16));
    auto v_buf = backend_.allocate_buffer(static_cast<size_t>(T) * nkv * hd * sizeof(__nv_bfloat16));
    auto attn_out = backend_.allocate_buffer(static_cast<size_t>(T) * attn_dim * sizeof(__nv_bfloat16));
    auto gate = backend_.allocate_buffer(static_cast<size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    auto up = backend_.allocate_buffer(static_cast<size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    auto ffn_act = backend_.allocate_buffer(static_cast<size_t>(T) * d_ff * sizeof(__nv_bfloat16));

    ASSERT_TRUE(cuda_check(cudaMemcpyAsync(
        hidden_a->data(), input_embeds->data(),
        static_cast<size_t>(T) * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream_)));

    __nv_bfloat16* hidden = static_cast<__nv_bfloat16*>(hidden_a->data());
    __nv_bfloat16* next_hidden = static_cast<__nv_bfloat16*>(hidden_b->data());

    std::string out_dir = dir_ + "/our_layer_outputs";
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    int ret = system(mkdir_cmd.c_str());
    (void)ret;

    auto dump_bf16_to_f32 = [&](const char* name, const __nv_bfloat16* src, size_t n) {
        cudaStreamSynchronize(stream_);
        std::vector<__nv_bfloat16> tmp(n);
        std::vector<float> f32(n);
        cudaMemcpy(tmp.data(), src, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; ++i) f32[i] = __bfloat162float(tmp[i]);
        save_bin(out_dir + "/" + name, f32.data(), n);
    };

    std::vector<float> host_hidden(static_cast<size_t>(T) * D);
    std::vector<__nv_bfloat16> host_bf16(static_cast<size_t>(T) * D);

    // Save input embeddings (layer 0 input)
    dump_bf16_to_f32("embedding.bin", hidden, static_cast<size_t>(T) * D);

    // Load per-layer weights
    for (int l = 0; l < n_layers; ++l) {
        bool is_first_layer = (l == 0);
        const std::string p = "model.layers." + std::to_string(l);

        auto rms_attn = loader_->load<__nv_bfloat16>(backend_, p + ".input_layernorm.weight", {D}); ASSERT_TRUE(rms_attn);
        auto qkv_w = loader_->load<__nv_bfloat16>(backend_, p + ".self_attn.q_proj.weight", {nq * hd, D}); ASSERT_TRUE(qkv_w);
        auto k_w = loader_->load<__nv_bfloat16>(backend_, p + ".self_attn.k_proj.weight", {nkv * hd, D}); ASSERT_TRUE(k_w);
        auto v_w = loader_->load<__nv_bfloat16>(backend_, p + ".self_attn.v_proj.weight", {nkv * hd, D}); ASSERT_TRUE(v_w);
        auto o_w = loader_->load<__nv_bfloat16>(backend_, p + ".self_attn.o_proj.weight", {D, nq * hd}); ASSERT_TRUE(o_w);
        auto rms_ffn = loader_->load<__nv_bfloat16>(backend_, p + ".post_attention_layernorm.weight", {D}); ASSERT_TRUE(rms_ffn);
        auto gate_w = loader_->load<__nv_bfloat16>(backend_, p + ".mlp.gate_proj.weight", {d_ff, D}); ASSERT_TRUE(gate_w);
        auto up_w = loader_->load<__nv_bfloat16>(backend_, p + ".mlp.up_proj.weight", {d_ff, D}); ASSERT_TRUE(up_w);
        auto down_w = loader_->load<__nv_bfloat16>(backend_, p + ".mlp.down_proj.weight", {D, d_ff}); ASSERT_TRUE(down_w);

        // Merge QKV: Q weight [nq*hd, D], K weight [nkv*hd, D], V weight [nkv*hd, D]
        const size_t q_elems = static_cast<size_t>(nq * hd) * D;
        const size_t kv_elems = static_cast<size_t>(nkv * hd) * D;
        auto qkv_merged = backend_.allocate_buffer((q_elems + 2 * kv_elems) * sizeof(__nv_bfloat16));
        cudaMemcpy(qkv_merged->data(), (*qkv_w)->data(), q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<__nv_bfloat16*>(qkv_merged->data()) + q_elems, (*k_w)->data(), kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<__nv_bfloat16*>(qkv_merged->data()) + q_elems + kv_elems, (*v_w)->data(), kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

        // QK norm weights
        std::unique_ptr<DeviceBuffer> q_norm_w, k_norm_w;
        auto qn = loader_->load<__nv_bfloat16>(backend_, p + ".self_attn.q_norm.weight", {hd});
        if (qn) q_norm_w = std::move(*qn);
        auto kn = loader_->load<__nv_bfloat16>(backend_, p + ".self_attn.k_norm.weight", {hd});
        if (kn) k_norm_w = std::move(*kn);

        // ---- Attention block ----
        ASSERT_TRUE(backend_.rms_norm(RmsNormParams{
            .input_ = hidden, .weight_ = static_cast<__nv_bfloat16*>((*rms_attn)->data()), .output_ = static_cast<__nv_bfloat16*>(normed->data()),
            .rows_ = T, .dim_ = D, .eps_ = eps, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_attn_norm.bin", static_cast<__nv_bfloat16*>(normed->data()), static_cast<size_t>(T) * D);

        ASSERT_TRUE(backend_.gemm(GemmParams{
            .a_ = static_cast<__nv_bfloat16*>(qkv_merged->data()), .b_ = static_cast<__nv_bfloat16*>(normed->data()), .c_ = static_cast<__nv_bfloat16*>(qkv_out->data()),
            .m_ = qkv_dim, .n_ = T, .k_ = D,
            .lda_ = D, .ldb_ = D, .ldc_ = qkv_dim,
            .trans_a_ = true, .trans_b_ = false, .stream_ = stream_,
        }));

        ASSERT_TRUE(backend_.split_qkv(SplitQkvParams{
            .qkv_ = static_cast<__nv_bfloat16*>(qkv_out->data()), .q_ = static_cast<__nv_bfloat16*>(q_buf->data()), .k_ = static_cast<__nv_bfloat16*>(k_buf->data()), .v_ = static_cast<__nv_bfloat16*>(v_buf->data()),
            .num_tokens_ = T, .num_q_heads_ = nq, .num_kv_heads_ = nkv, .head_dim_ = hd, .stream_ = stream_,
        }));
        if (is_first_layer) {
            dump_bf16_to_f32("l0_q.bin", static_cast<__nv_bfloat16*>(q_buf->data()), static_cast<size_t>(T) * nq * hd);
            dump_bf16_to_f32("l0_k.bin", static_cast<__nv_bfloat16*>(k_buf->data()), static_cast<size_t>(T) * nkv * hd);
            dump_bf16_to_f32("l0_v.bin", static_cast<__nv_bfloat16*>(v_buf->data()), static_cast<size_t>(T) * nkv * hd);
        }

        if (q_norm_w) {
            ASSERT_TRUE(backend_.rms_norm(RmsNormParams{
                .input_ = static_cast<__nv_bfloat16*>(q_buf->data()), .weight_ = static_cast<__nv_bfloat16*>(q_norm_w->data()), .output_ = static_cast<__nv_bfloat16*>(q_buf->data()),
                .rows_ = T * nq, .dim_ = hd, .eps_ = eps, .stream_ = stream_,
            }));
        }
        if (k_norm_w) {
            ASSERT_TRUE(backend_.rms_norm(RmsNormParams{
                .input_ = static_cast<__nv_bfloat16*>(k_buf->data()), .weight_ = static_cast<__nv_bfloat16*>(k_norm_w->data()), .output_ = static_cast<__nv_bfloat16*>(k_buf->data()),
                .rows_ = T * nkv, .dim_ = hd, .eps_ = eps, .stream_ = stream_,
            }));
        }
        if (is_first_layer) {
            dump_bf16_to_f32("l0_q_normed.bin", static_cast<__nv_bfloat16*>(q_buf->data()), static_cast<size_t>(T) * nq * hd);
            dump_bf16_to_f32("l0_k_normed.bin", static_cast<__nv_bfloat16*>(k_buf->data()), static_cast<size_t>(T) * nkv * hd);
        }

        ASSERT_TRUE(backend_.rope(RopeParams{
            .q_ = static_cast<__nv_bfloat16*>(q_buf->data()), .k_ = static_cast<__nv_bfloat16*>(k_buf->data()), .positions_ = static_cast<int32_t*>(pos_buf->data()),
            .rope_cache_ = rope_cache.data(), .num_tokens_ = T, .num_q_heads_ = nq,
            .num_kv_heads_ = nkv, .head_dim_ = hd, .rotary_dim_ = hd,
            .max_position_ = rope_cache.max_position(), .stream_ = stream_,
        }));

        ASSERT_TRUE(backend_.naive_attention(NaiveAttnParams{
            .q_ = static_cast<__nv_bfloat16*>(q_buf->data()), .k_ = static_cast<__nv_bfloat16*>(k_buf->data()), .v_ = static_cast<__nv_bfloat16*>(v_buf->data()), .output_ = static_cast<__nv_bfloat16*>(attn_out->data()),
            .num_tokens_ = T, .num_q_heads_ = nq, .num_kv_heads_ = nkv, .head_dim_ = hd, .stream_ = stream_,
        }));

        ASSERT_TRUE(backend_.gemm(GemmParams{
            .a_ = static_cast<__nv_bfloat16*>((*o_w)->data()), .b_ = static_cast<__nv_bfloat16*>(attn_out->data()), .c_ = next_hidden,
            .m_ = D, .n_ = T, .k_ = attn_dim,
            .lda_ = attn_dim, .ldb_ = attn_dim, .ldc_ = D,
            .trans_a_ = true, .trans_b_ = false, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_o_proj.bin", next_hidden, static_cast<size_t>(T) * D);

        ASSERT_TRUE(backend_.element_add(ElementAddParams{
            .dst_ = next_hidden, .src_ = hidden, .n_ = static_cast<int64_t>(T) * D, .stream_ = stream_,
        }));
        std::swap(hidden, next_hidden);
        if (is_first_layer)
            dump_bf16_to_f32("l0_attn_residual.bin", hidden, static_cast<size_t>(T) * D);

        // ---- FFN block ----
        ASSERT_TRUE(backend_.rms_norm(RmsNormParams{
            .input_ = hidden, .weight_ = static_cast<__nv_bfloat16*>((*rms_ffn)->data()), .output_ = static_cast<__nv_bfloat16*>(normed->data()),
            .rows_ = T, .dim_ = D, .eps_ = eps, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_ffn_norm.bin", static_cast<__nv_bfloat16*>(normed->data()), static_cast<size_t>(T) * D);

        ASSERT_TRUE(backend_.gemm(GemmParams{
            .a_ = static_cast<__nv_bfloat16*>((*gate_w)->data()), .b_ = static_cast<__nv_bfloat16*>(normed->data()), .c_ = static_cast<__nv_bfloat16*>(gate->data()),
            .m_ = d_ff, .n_ = T, .k_ = D,
            .lda_ = D, .ldb_ = D, .ldc_ = d_ff,
            .trans_a_ = true, .trans_b_ = false, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_gate.bin", static_cast<__nv_bfloat16*>(gate->data()), static_cast<size_t>(T) * d_ff);

        ASSERT_TRUE(backend_.gemm(GemmParams{
            .a_ = static_cast<__nv_bfloat16*>((*up_w)->data()), .b_ = static_cast<__nv_bfloat16*>(normed->data()), .c_ = static_cast<__nv_bfloat16*>(up->data()),
            .m_ = d_ff, .n_ = T, .k_ = D,
            .lda_ = D, .ldb_ = D, .ldc_ = d_ff,
            .trans_a_ = true, .trans_b_ = false, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_up.bin", static_cast<__nv_bfloat16*>(up->data()), static_cast<size_t>(T) * d_ff);

        ASSERT_TRUE(backend_.silu_mul(SiluMulParams{
            .gate_ = static_cast<__nv_bfloat16*>(gate->data()), .up_ = static_cast<__nv_bfloat16*>(up->data()), .output_ = static_cast<__nv_bfloat16*>(ffn_act->data()),
            .n_ = static_cast<int64_t>(T) * d_ff, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_silu_mul.bin", static_cast<__nv_bfloat16*>(ffn_act->data()), static_cast<size_t>(T) * d_ff);

        ASSERT_TRUE(backend_.gemm(GemmParams{
            .a_ = static_cast<__nv_bfloat16*>((*down_w)->data()), .b_ = static_cast<__nv_bfloat16*>(ffn_act->data()), .c_ = next_hidden,
            .m_ = D, .n_ = T, .k_ = d_ff,
            .lda_ = d_ff, .ldb_ = d_ff, .ldc_ = D,
            .trans_a_ = true, .trans_b_ = false, .stream_ = stream_,
        }));
        if (is_first_layer)
            dump_bf16_to_f32("l0_down.bin", next_hidden, static_cast<size_t>(T) * D);

        ASSERT_TRUE(backend_.element_add(ElementAddParams{
            .dst_ = next_hidden, .src_ = hidden, .n_ = static_cast<int64_t>(T) * D, .stream_ = stream_,
        }));
        std::swap(hidden, next_hidden);

        // Dump after this layer
        cudaStreamSynchronize(stream_);
        cudaMemcpy(host_bf16.data(), hidden, T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        for (int i = 0; i < T * D; ++i)
            host_hidden[static_cast<size_t>(i)] = __bfloat162float(host_bf16[static_cast<size_t>(i)]);
        save_bin(out_dir + "/layer_" + std::to_string(l) + ".bin",
                 host_hidden.data(), static_cast<size_t>(T) * D);
    }

    // Final norm
    ASSERT_TRUE(backend_.rms_norm(RmsNormParams{
        .input_ = hidden, .weight_ = static_cast<__nv_bfloat16*>((*rms_final)->data()), .output_ = static_cast<__nv_bfloat16*>(normed->data()),
        .rows_ = T, .dim_ = D, .eps_ = eps, .stream_ = stream_,
    }));
    cudaStreamSynchronize(stream_);
    cudaMemcpy(host_bf16.data(), normed->data(), T * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < T * D; ++i)
        host_hidden[static_cast<size_t>(i)] = __bfloat162float(host_bf16[static_cast<size_t>(i)]);
    save_bin(out_dir + "/final_norm.bin", host_hidden.data(), static_cast<size_t>(T) * D);

    printf("Dumped layer outputs to %s\n", out_dir.c_str());
}
