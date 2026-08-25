#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "backend/backend.h"
#include "backend/cuda/cuda_utils.h"
#include "cache/block.h"
#include "cache/kv_cache_manager.h"
#include "cache/kv_cache_storage.h"
#include "model/config.h"
#include "model/loader.h"
#include "model/rope/rope_cache.h"
#include "facade/ops.h"
#include "tokenizer/byte_level_bpe_tokenizer.h"

using namespace ccinfer;

namespace {

template <typename B>
std::shared_ptr<Buffer> alloc_buf(B& backend, size_t bytes) {
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

void save_bin(const std::string& path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(float)));
}

void dump_bf16_to_f32(const std::string& dir, const char* name, const Tensor& src, size_t n) {
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> tmp(n);
    std::vector<float> f32(n);
    cudaMemcpy(tmp.data(), src.data(), n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) f32[i] = __bfloat162float(tmp[i]);
    save_bin(dir + "/" + name, f32.data(), n);
}

}  // namespace

class LayerMatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!model_available()) GTEST_SKIP() << "CCINFER_TEST_MODEL_DIR not set";
        cudaStreamCreate(&stream_);
        auto b = Backend::create(0);
        if (!b) GTEST_SKIP() << "CUDA unavailable";
        backend_ = std::move(*b);
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
        if (stream_) {
            cudaDeviceSynchronize();
            cudaStreamDestroy(stream_);
        }
    }

    std::string dir_;
    ModelConfig config_;
    ByteLevelBpeTokenizer tokenizer_;
    std::unique_ptr<WeightLoader> loader_;
    std::unique_ptr<Backend> backend_;
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
    constexpr ccop::Device kCuda0{ccop::DeviceType::kCUDA, 0};
    const ccop::ExecutionContext ctx{stream_, backend_->context().blas_handle};

    auto embed =
        loader_->load_tensor<__nv_bfloat16>(*backend_, "model.embed_tokens.weight", {V, D});
    ASSERT_TRUE(embed);
    auto rms_final = loader_->load_tensor<__nv_bfloat16>(*backend_, "model.norm.weight", {D});
    ASSERT_TRUE(rms_final);
    auto lm_head = loader_->load_tensor<__nv_bfloat16>(*backend_, "lm_head.weight", {V, D});
    ASSERT_TRUE(lm_head);

    auto rc = RopeCache::create(config_.max_seq_len_, hd, config_.rope_theta_, *backend_);
    ASSERT_TRUE(rc);
    auto& rope_cache = *rc;

    auto pos_buf = alloc_buf(*backend_, static_cast<size_t>(T) * sizeof(std::int32_t));
    {
        std::vector<std::int32_t> pos_host(static_cast<size_t>(T));
        for (int i = 0; i < T; ++i) pos_host[static_cast<size_t>(i)] = i;
        cudaMemcpy(pos_buf->data(), pos_host.data(), T * sizeof(std::int32_t),
                   cudaMemcpyHostToDevice);
    }
    auto token_ids_buf = alloc_buf(*backend_, static_cast<size_t>(T) * sizeof(std::int32_t));
    cudaMemcpy(token_ids_buf->data(), tokens.data(), T * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    auto input_embeds_buf =
        alloc_buf(*backend_, static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    Tensor input_embeds(input_embeds_buf, ccop::DType::kBFloat16, {T, D});
    Tensor embed_tensor = std::move(*embed);
    Tensor token_ids(token_ids_buf, ccop::DType::kInt32, {T});
    ASSERT_TRUE(
        map_result(ccop::embed(embed_tensor, token_ids, &input_embeds, ctx)).has_value());

    const size_t bf16_size = ccop::dtype_size(ccop::DType::kBFloat16);
    auto hidden_a_buf = alloc_buf(*backend_, static_cast<size_t>(T) * D * bf16_size);
    auto hidden_b_buf = alloc_buf(*backend_, static_cast<size_t>(T) * D * bf16_size);
    auto normed_buf = alloc_buf(*backend_, static_cast<size_t>(T) * D * bf16_size);
    auto qkv_out_buf = alloc_buf(*backend_, static_cast<size_t>(T) * qkv_dim * bf16_size);
    auto q_buf = alloc_buf(*backend_, static_cast<size_t>(T) * nq * hd * bf16_size);
    auto k_buf = alloc_buf(*backend_, static_cast<size_t>(T) * nkv * hd * bf16_size);
    auto v_buf = alloc_buf(*backend_, static_cast<size_t>(T) * nkv * hd * bf16_size);
    auto attn_out_buf = alloc_buf(*backend_, static_cast<size_t>(T) * attn_dim * bf16_size);
    auto gate_buf = alloc_buf(*backend_, static_cast<size_t>(T) * d_ff * bf16_size);
    auto up_buf = alloc_buf(*backend_, static_cast<size_t>(T) * d_ff * bf16_size);
    auto ffn_act_buf = alloc_buf(*backend_, static_cast<size_t>(T) * d_ff * bf16_size);

    const int max_blocks = std::max(1, (T + kKVBlockSize - 1) / kKVBlockSize);
    auto kv_mgr = std::make_unique<KVCacheManager>();
    {
        auto storage = KVCacheStorage::create(*backend_, n_layers, max_blocks, kKVBlockSize, nkv,
                                              hd, ccop::DType::kBFloat16);
        ASSERT_TRUE(storage.has_value());
        ASSERT_TRUE(kv_mgr->init(std::move(*storage), max_blocks, kKVBlockSize));
        auto blocks = kv_mgr->allocate_blocks(max_blocks);
        ASSERT_TRUE(blocks.has_value());
    }

    std::vector<std::int32_t> slot_mapping_host(static_cast<size_t>(T));
    for (int i = 0; i < T; ++i) slot_mapping_host[static_cast<size_t>(i)] = i;
    auto slot_mapping_buf = alloc_buf(*backend_, static_cast<size_t>(T) * sizeof(std::int32_t));
    cudaMemcpy(slot_mapping_buf->data(), slot_mapping_host.data(), T * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    std::vector<std::int32_t> bt_host(static_cast<size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i) bt_host[static_cast<size_t>(i)] = i;
    auto block_table_buf =
        alloc_buf(*backend_, static_cast<size_t>(max_blocks) * sizeof(std::int32_t));
    cudaMemcpy(block_table_buf->data(), bt_host.data(), max_blocks * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    std::vector<std::int32_t> qsl_host{0, T};
    auto qsl_buf = alloc_buf(*backend_, 2 * sizeof(std::int32_t));
    cudaMemcpy(qsl_buf->data(), qsl_host.data(), 2 * sizeof(std::int32_t), cudaMemcpyHostToDevice);
    std::vector<std::int32_t> ctx_host{T};
    auto ctx_buf = alloc_buf(*backend_, sizeof(std::int32_t));
    cudaMemcpy(ctx_buf->data(), ctx_host.data(), sizeof(std::int32_t), cudaMemcpyHostToDevice);

    Tensor hidden(hidden_a_buf, ccop::DType::kBFloat16, {T, D});
    Tensor hidden_flat(hidden_a_buf, ccop::DType::kBFloat16, {static_cast<std::int64_t>(T) * D});
    Tensor next_hidden(hidden_b_buf, ccop::DType::kBFloat16, {T, D});
    Tensor next_hidden_flat(hidden_b_buf, ccop::DType::kBFloat16,
                            {static_cast<std::int64_t>(T) * D});
    Tensor normed(normed_buf, ccop::DType::kBFloat16, {T, D});
    Tensor qkv_out(qkv_out_buf, ccop::DType::kBFloat16, {T, qkv_dim});
    Tensor q(q_buf, ccop::DType::kBFloat16, {T, nq, hd});
    Tensor q_norm_2d(q_buf, ccop::DType::kBFloat16, {static_cast<std::int64_t>(T) * nq, hd});
    Tensor k(k_buf, ccop::DType::kBFloat16, {T, nkv, hd});
    Tensor k_norm_2d(k_buf, ccop::DType::kBFloat16, {static_cast<std::int64_t>(T) * nkv, hd});
    Tensor v(v_buf, ccop::DType::kBFloat16, {T, nkv, hd});
    Tensor attn_out(attn_out_buf, ccop::DType::kBFloat16, {T, attn_dim});
    Tensor attn_out_3d(attn_out_buf, ccop::DType::kBFloat16, {T, nq, hd});
    Tensor gate(gate_buf, ccop::DType::kBFloat16, {T, d_ff});
    Tensor gate_flat(gate_buf, ccop::DType::kBFloat16, {static_cast<std::int64_t>(T) * d_ff});
    Tensor up(up_buf, ccop::DType::kBFloat16, {T, d_ff});
    Tensor up_flat(up_buf, ccop::DType::kBFloat16, {static_cast<std::int64_t>(T) * d_ff});
    Tensor ffn_act(ffn_act_buf, ccop::DType::kBFloat16, {T, d_ff});
    Tensor ffn_act_flat(ffn_act_buf, ccop::DType::kBFloat16, {static_cast<std::int64_t>(T) * d_ff});
    Tensor positions(pos_buf, ccop::DType::kInt32, {T});
    Tensor slot_mapping(slot_mapping_buf, ccop::DType::kInt32, {T});
    Tensor block_table(block_table_buf, ccop::DType::kInt32, {1, max_blocks});
    Tensor query_start_loc(qsl_buf, ccop::DType::kInt32, {2});
    Tensor context_lens(ctx_buf, ccop::DType::kInt32, {1});

    auto sync =
        cudaMemcpyAsync(hidden.data(), input_embeds.data(), static_cast<size_t>(T) * D * bf16_size,
                        cudaMemcpyDeviceToDevice, stream_);
    ASSERT_TRUE(cuda_check(sync).has_value());

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::string out_dir = dir_ + "/our_layer_outputs";
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    int ret = system(mkdir_cmd.c_str());
    (void)ret;

    dump_bf16_to_f32(out_dir, "embedding.bin", hidden, static_cast<size_t>(T) * D);

    std::vector<float> host_hidden(static_cast<size_t>(T) * D);
    std::vector<__nv_bfloat16> host_bf16(static_cast<size_t>(T) * D);

    for (int l = 0; l < n_layers; ++l) {
        const bool is_first_layer = (l == 0);
        const std::string p = "model.layers." + std::to_string(l);

        auto rms_attn =
            loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".input_layernorm.weight", {D});
        ASSERT_TRUE(rms_attn);
        auto qkv_w = loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".self_attn.q_proj.weight",
                                                         {nq * hd, D});
        auto k_w = loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".self_attn.k_proj.weight",
                                                       {nkv * hd, D});
        auto v_w = loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".self_attn.v_proj.weight",
                                                       {nkv * hd, D});
        auto o_w = loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".self_attn.o_proj.weight",
                                                       {D, nq * hd});
        auto rms_ffn = loader_->load_tensor<__nv_bfloat16>(
            *backend_, p + ".post_attention_layernorm.weight", {D});
        auto gate_w =
            loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".mlp.gate_proj.weight", {d_ff, D});
        auto up_w =
            loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".mlp.up_proj.weight", {d_ff, D});
        auto down_w =
            loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".mlp.down_proj.weight", {D, d_ff});
        ASSERT_TRUE(qkv_w && k_w && v_w && o_w && rms_ffn && gate_w && up_w && down_w);

        const size_t q_elems = static_cast<size_t>(nq * hd) * D;
        const size_t kv_elems = static_cast<size_t>(nkv * hd) * D;
        auto qkv_merged_buf =
            alloc_buf(*backend_, (q_elems + 2 * kv_elems) * sizeof(__nv_bfloat16));
        cudaMemcpy(qkv_merged_buf->data(), qkv_w->data(), q_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<__nv_bfloat16*>(qkv_merged_buf->data()) + q_elems, k_w->data(),
                   kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        cudaMemcpy(static_cast<__nv_bfloat16*>(qkv_merged_buf->data()) + q_elems + kv_elems,
                   v_w->data(), kv_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
        Tensor qkv_weight(qkv_merged_buf, ccop::DType::kBFloat16, {qkv_dim, D});

        std::shared_ptr<Buffer> q_norm_buf, k_norm_buf;
        auto qn =
            loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".self_attn.q_norm.weight", {hd});
        if (qn) q_norm_buf = qn->buffer();
        auto kn =
            loader_->load_tensor<__nv_bfloat16>(*backend_, p + ".self_attn.k_norm.weight", {hd});
        if (kn) k_norm_buf = kn->buffer();

        ASSERT_TRUE(
            map_result(ccop::rms_norm(&normed, hidden, *rms_attn, eps, ctx)).has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_attn_norm.bin", normed, static_cast<size_t>(T) * D);

        ASSERT_TRUE(
            map_result(ccop::gemm(&qkv_out, normed, qkv_weight, false, true, 1.0f, 0.0f, ctx))
                .has_value());
        ASSERT_TRUE(map_result(ccop::split_qkv(qkv_out, &q, &k, &v, ctx)).has_value());
        if (is_first_layer) {
            dump_bf16_to_f32(out_dir, "l0_q.bin", q, static_cast<size_t>(T) * nq * hd);
            dump_bf16_to_f32(out_dir, "l0_k.bin", k, static_cast<size_t>(T) * nkv * hd);
            dump_bf16_to_f32(out_dir, "l0_v.bin", v, static_cast<size_t>(T) * nkv * hd);
        }

        if (q_norm_buf) {
            ASSERT_TRUE(
                map_result(ccop::rms_norm(&q_norm_2d, q_norm_2d,
                                              Tensor(q_norm_buf, ccop::DType::kBFloat16, {hd}), eps,
                                              ctx))
                    .has_value());
        }
        if (k_norm_buf) {
            ASSERT_TRUE(
                map_result(ccop::rms_norm(&k_norm_2d, k_norm_2d,
                                              Tensor(k_norm_buf, ccop::DType::kBFloat16, {hd}), eps,
                                              ctx))
                    .has_value());
        }
        if (is_first_layer) {
            dump_bf16_to_f32(out_dir, "l0_q_normed.bin", q, static_cast<size_t>(T) * nq * hd);
            dump_bf16_to_f32(out_dir, "l0_k_normed.bin", k, static_cast<size_t>(T) * nkv * hd);
        }

        ASSERT_TRUE(map_result(ccop::rope(&q, &k, positions, rope_cache.tensor(), hd, ctx))
                        .has_value());

        {
            auto k_cache = kv_mgr->k_cache(l);
            auto v_cache = kv_mgr->v_cache(l);
            ASSERT_TRUE(
                map_result(ccop::write_kv_cache(k, v, &k_cache, &v_cache, slot_mapping, ctx))
                    .has_value());
        }
        {
            auto k_blocks = kv_mgr->k_cache_blocks(l);
            auto v_blocks = kv_mgr->v_cache_blocks(l);
            ASSERT_TRUE(map_result(ccop::prefill_attention(q, k_blocks, v_blocks, block_table,
                                                               query_start_loc, context_lens,
                                                               &attn_out_3d, attn_scale, ctx))
                            .has_value());
        }

        ASSERT_TRUE(
            map_result(ccop::gemm(&next_hidden, attn_out, *o_w, false, true, 1.0f, 0.0f, ctx))
                .has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_o_proj.bin", next_hidden, static_cast<size_t>(T) * D);

        ASSERT_TRUE(
            map_result(ccop::element_add(&next_hidden_flat, hidden_flat, ctx)).has_value());
        std::swap(hidden, next_hidden);
        std::swap(hidden_flat, next_hidden_flat);
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_attn_residual.bin", hidden, static_cast<size_t>(T) * D);

        ASSERT_TRUE(
            map_result(ccop::rms_norm(&normed, hidden, *rms_ffn, eps, ctx)).has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_ffn_norm.bin", normed, static_cast<size_t>(T) * D);

        ASSERT_TRUE(map_result(ccop::gemm(&gate, normed, *gate_w, false, true, 1.0f, 0.0f, ctx))
                        .has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_gate.bin", gate, static_cast<size_t>(T) * d_ff);

        ASSERT_TRUE(map_result(ccop::gemm(&up, normed, *up_w, false, true, 1.0f, 0.0f, ctx))
                        .has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_up.bin", up, static_cast<size_t>(T) * d_ff);

        ASSERT_TRUE(
            map_result(ccop::silu_mul(&ffn_act_flat, gate_flat, up_flat, ctx)).has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_silu_mul.bin", ffn_act, static_cast<size_t>(T) * d_ff);

        ASSERT_TRUE(
            map_result(ccop::gemm(&next_hidden, ffn_act, *down_w, false, true, 1.0f, 0.0f, ctx))
                .has_value());
        if (is_first_layer)
            dump_bf16_to_f32(out_dir, "l0_down.bin", next_hidden, static_cast<size_t>(T) * D);

        ASSERT_TRUE(
            map_result(ccop::element_add(&next_hidden_flat, hidden_flat, ctx)).has_value());
        std::swap(hidden, next_hidden);
        std::swap(hidden_flat, next_hidden_flat);

        cudaDeviceSynchronize();
        cudaMemcpy(host_bf16.data(), hidden.data(),
                   static_cast<size_t>(T) * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        for (int i = 0; i < T * D; ++i)
            host_hidden[static_cast<size_t>(i)] =
                __bfloat162float(host_bf16[static_cast<size_t>(i)]);
        save_bin(out_dir + "/layer_" + std::to_string(l) + ".bin", host_hidden.data(),
                 static_cast<size_t>(T) * D);
    }

    ASSERT_TRUE(map_result(ccop::rms_norm(&normed, hidden, *rms_final, eps, ctx)).has_value());
    cudaDeviceSynchronize();
    cudaMemcpy(host_bf16.data(), normed.data(), static_cast<size_t>(T) * D * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < T * D; ++i)
        host_hidden[static_cast<size_t>(i)] = __bfloat162float(host_bf16[static_cast<size_t>(i)]);
    save_bin(out_dir + "/final_norm.bin", host_hidden.data(), static_cast<size_t>(T) * D);

    printf("Dumped layer outputs to %s\n", out_dir.c_str());
}
