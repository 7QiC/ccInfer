#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "checkpoint/checkpoint.h"

namespace ccinfer {
namespace {

void write_string(std::ofstream& f, const std::string& s) {
    const uint64_t len = static_cast<uint64_t>(s.size());
    f.write(reinterpret_cast<const char*>(&len), sizeof(len));
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}
void write_u32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void write_u64(std::ofstream& f, uint64_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

std::vector<uint8_t> scalar_u32(uint32_t v) {
    std::vector<uint8_t> bytes(sizeof(v));
    std::memcpy(bytes.data(), &v, sizeof(v));
    return bytes;
}
std::vector<uint8_t> scalar_f32(float v) {
    std::vector<uint8_t> bytes(sizeof(v));
    std::memcpy(bytes.data(), &v, sizeof(v));
    return bytes;
}
std::vector<uint8_t> scalar_string(const std::string& s) {
    std::vector<uint8_t> bytes;
    const uint64_t len = static_cast<uint64_t>(s.size());
    const uint8_t* lenp = reinterpret_cast<const uint8_t*>(&len);
    bytes.insert(bytes.end(), lenp, lenp + sizeof(len));
    bytes.insert(bytes.end(), s.begin(), s.end());
    return bytes;
}

struct MetadataEntry {
    std::string key;
    uint32_t type;
    std::vector<uint8_t> value;
};

void write_hf_checkpoint(const std::string& dir) {
    std::filesystem::create_directories(dir);

    std::ofstream cfg(dir + "/config.json");
    cfg << R"({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "max_position_embeddings": 4096
    })";
    cfg.close();

    const std::string header = R"({
        "tensor": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}
    })";
    std::ofstream st(dir + "/model.safetensors", std::ios::binary);
    const uint64_t header_len = header.size();
    st.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    st.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::vector<uint8_t> data(8, 0x77);
    st.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    st.close();
}

void write_gguf_qwen35(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 1);
    write_u64(f, 1 + 16);  // arch + 16 qwen35 metadata

    const auto arch = scalar_string("qwen35");
    write_string(f, "general.architecture");
    write_u32(f, 8);
    f.write(reinterpret_cast<const char*>(arch.data()), static_cast<std::streamsize>(arch.size()));

    std::vector<MetadataEntry> entries = {
        {"qwen35.block_count", 4, scalar_u32(25)},
        {"qwen35.embedding_length", 4, scalar_u32(2048)},
        {"qwen35.feed_forward_length", 4, scalar_u32(6144)},
        {"qwen35.attention.head_count", 4, scalar_u32(8)},
        {"qwen35.attention.head_count_kv", 4, scalar_u32(2)},
        {"qwen35.attention.key_length", 4, scalar_u32(256)},
        {"qwen35.attention.layer_norm_rms_epsilon", 6, scalar_f32(1e-6f)},
        {"qwen35.context_length", 4, scalar_u32(262144)},
        {"qwen35.full_attention_interval", 4, scalar_u32(4)},
        {"qwen35.nextn_predict_layers", 4, scalar_u32(1)},
        {"qwen35.ssm.conv_kernel", 4, scalar_u32(4)},
        {"qwen35.ssm.state_size", 4, scalar_u32(128)},
        {"qwen35.ssm.group_count", 4, scalar_u32(16)},
        {"qwen35.ssm.time_step_rank", 4, scalar_u32(16)},
        {"qwen35.ssm.inner_size", 4, scalar_u32(2048)},
        {"qwen35.rope.freq_base", 6, scalar_f32(10000000.0f)},
    };
    for (const auto& e : entries) {
        write_string(f, e.key);
        write_u32(f, e.type);
        f.write(reinterpret_cast<const char*>(e.value.data()), static_cast<std::streamsize>(e.value.size()));
    }

    write_string(f, "token_embd.weight");
    write_u32(f, 2);
    write_u64(f, 2048);
    write_u64(f, 32);
    write_u32(f, 8);  // Q8_0
    write_u64(f, 0);
    f.close();
}

TEST(CheckpointTest, OpensHFDirectory) {
    const auto dir = std::filesystem::temp_directory_path() / "ccinfer-test-checkpoint-hf";
    std::filesystem::remove_all(dir);
    write_hf_checkpoint(dir.string());
    const auto cleanup = [&] { std::filesystem::remove_all(dir); };

    auto cp = Checkpoint::open(dir.string());
    ASSERT_TRUE(cp.has_value());

    auto cfg = (*cp)->load_config();
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->arch_, ModelArch::Qwen3);
    EXPECT_EQ(cfg->n_layers_, 28);

    auto& ws = (*cp)->weights();
    EXPECT_TRUE(ws.has("tensor"));
    auto raw = ws.read("tensor");
    ASSERT_TRUE(raw.has_value());
    EXPECT_EQ(raw->size(), 8u);

    cleanup();
}

TEST(CheckpointTest, OpensGgufFile) {
    const auto path = std::filesystem::temp_directory_path() / "ccinfer-test-checkpoint.gguf";
    std::filesystem::remove(path);
    write_gguf_qwen35(path.string());

    auto cp = Checkpoint::open(path.string());
    std::filesystem::remove(path);
    ASSERT_TRUE(cp.has_value());

    auto cfg = (*cp)->load_config();
    ASSERT_TRUE(cfg.has_value());
    EXPECT_EQ(cfg->arch_, ModelArch::Qwen3_5);
    EXPECT_EQ(cfg->n_layers_, 24);
    EXPECT_EQ(cfg->nextn_predict_layers_, 1);

    auto& ws = (*cp)->weights();
    EXPECT_TRUE(ws.has("token_embd.weight"));
    EXPECT_TRUE(ws.info("token_embd.weight").has_value());
}

TEST(CheckpointTest, RejectsUnsupportedPath) {
    EXPECT_FALSE(Checkpoint::open("/tmp/does-not-exist-ccinfer").has_value());

    const auto empty_dir = std::filesystem::temp_directory_path() / "ccinfer-test-checkpoint-empty";
    std::filesystem::remove_all(empty_dir);
    std::filesystem::create_directories(empty_dir);
    EXPECT_FALSE(Checkpoint::open(empty_dir.string()).has_value());
    std::filesystem::remove_all(empty_dir);
}

}  // namespace
}  // namespace ccinfer
