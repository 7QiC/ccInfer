#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "checkpoint/gguf/config_loader.h"
#include "checkpoint/gguf/reader.h"

namespace ccinfer {
namespace {

class ScopedTempFile {
public:
    explicit ScopedTempFile(const std::string& name) : path_(std::filesystem::temp_directory_path() / name) {
        std::filesystem::remove(path_);
    }
    ~ScopedTempFile() { std::filesystem::remove(path_); }

    const std::string& path() const { return path_; }
    std::string path_;
};

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

void write_gguf(const std::string& path, const std::string& arch,
                std::vector<MetadataEntry> extra) {
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 1);  // token_embd
    write_u64(f, static_cast<uint64_t>(extra.size() + 1));  // metadata

    write_string(f, "general.architecture");
    write_u32(f, 8);
    auto arch_bytes = scalar_string(arch);
    f.write(reinterpret_cast<const char*>(arch_bytes.data()), static_cast<std::streamsize>(arch_bytes.size()));

    for (const auto& e : extra) {
        write_string(f, e.key);
        write_u32(f, e.type);
        f.write(reinterpret_cast<const char*>(e.value.data()), static_cast<std::streamsize>(e.value.size()));
    }

    write_string(f, "token_embd.weight");
    write_u32(f, 2);
    write_u64(f, 2048);
    write_u64(f, 100);
    write_u32(f, 8);  // Q8_0
    write_u64(f, 0);
    f.close();
}

std::vector<MetadataEntry> qwen35_metadata(int interval = 4, int nextn = 1) {
    return {
        {"qwen35.block_count", 4, scalar_u32(static_cast<uint32_t>(24 + nextn))},
        {"qwen35.embedding_length", 4, scalar_u32(2048)},
        {"qwen35.feed_forward_length", 4, scalar_u32(6144)},
        {"qwen35.attention.head_count", 4, scalar_u32(8)},
        {"qwen35.attention.head_count_kv", 4, scalar_u32(2)},
        {"qwen35.attention.key_length", 4, scalar_u32(256)},
        {"qwen35.attention.layer_norm_rms_epsilon", 6, scalar_f32(1e-6f)},
        {"qwen35.context_length", 4, scalar_u32(262144)},
        {"qwen35.full_attention_interval", 4, scalar_u32(static_cast<uint32_t>(interval))},
        {"qwen35.nextn_predict_layers", 4, scalar_u32(static_cast<uint32_t>(nextn))},
        {"qwen35.ssm.conv_kernel", 4, scalar_u32(4)},
        {"qwen35.ssm.state_size", 4, scalar_u32(128)},
        {"qwen35.ssm.group_count", 4, scalar_u32(16)},
        {"qwen35.ssm.time_step_rank", 4, scalar_u32(16)},
        {"qwen35.ssm.inner_size", 4, scalar_u32(2048)},
        {"qwen35.rope.freq_base", 6, scalar_f32(10000000.0f)},
    };
}

TEST(GgufConfigLoaderTest, LegalQwen35Config) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-config.gguf").path();
    write_gguf(path, "qwen35", qwen35_metadata(4, 1));
    auto reader_r = GGUFReader::create(path);
    ASSERT_TRUE(reader_r.has_value());

    auto cfg_r = GgufConfigLoader::load(**reader_r);
    ASSERT_TRUE(cfg_r.has_value());
    const auto& cfg = *cfg_r;

    EXPECT_EQ(cfg.arch_, ModelArch::Qwen3_5);
    EXPECT_EQ(cfg.n_layers_, 24);
    EXPECT_EQ(cfg.d_model_, 2048);
    EXPECT_EQ(cfg.d_ff_, 6144);
    EXPECT_EQ(cfg.n_q_heads_, 8);
    EXPECT_EQ(cfg.n_kv_heads_, 2);
    EXPECT_EQ(cfg.head_dim_, 256);
    EXPECT_EQ(cfg.vocab_size_, 100);
    EXPECT_EQ(cfg.max_seq_len_, 262144);
    EXPECT_EQ(cfg.full_attention_interval_, 4);
    EXPECT_EQ(cfg.nextn_predict_layers_, 1);
    EXPECT_EQ(cfg.ssm_conv_kernel_, 4);
    EXPECT_EQ(cfg.ssm_state_size_, 128);
    EXPECT_EQ(cfg.ssm_group_count_, 16);
    EXPECT_EQ(cfg.ssm_time_step_rank_, 16);
    EXPECT_EQ(cfg.ssm_inner_size_, 2048);

    ASSERT_EQ(cfg.layer_types_.size(), 25u);
    EXPECT_EQ(cfg.layer_types_[3], LayerType::FullAttention);
    EXPECT_EQ(cfg.layer_types_[4], LayerType::GatedDeltaNet);
    EXPECT_EQ(cfg.layer_types_[24], LayerType::MtpPredictor);
    EXPECT_EQ(static_cast<int>(std::count(cfg.layer_types_.begin(), cfg.layer_types_.end(),
                                          LayerType::FullAttention)), 6);
    EXPECT_EQ(static_cast<int>(std::count(cfg.layer_types_.begin(), cfg.layer_types_.end(),
                                          LayerType::MtpPredictor)), 1);
}

TEST(GgufConfigLoaderTest, RejectsNonQwen35Arch) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-config-badarch.gguf").path();
    write_gguf(path, "other", qwen35_metadata());
    auto reader_r = GGUFReader::create(path);
    ASSERT_TRUE(reader_r.has_value());

    auto cfg_r = GgufConfigLoader::load(**reader_r);
    EXPECT_FALSE(cfg_r.has_value());
    EXPECT_EQ(cfg_r.error(), ErrorCode::ModelUnsupportedArch);
}

TEST(GgufConfigLoaderTest, RejectsInvalidFullAttentionInterval) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-config-badinterval.gguf").path();
    write_gguf(path, "qwen35", qwen35_metadata(0, 1));
    auto reader_r = GGUFReader::create(path);
    ASSERT_TRUE(reader_r.has_value());

    auto cfg_r = GgufConfigLoader::load(**reader_r);
    EXPECT_FALSE(cfg_r.has_value());
    EXPECT_EQ(cfg_r.error(), ErrorCode::ModelConfigInvalid);
}

TEST(GgufConfigLoaderTest, NoNextnLayersProducesNoMtpPredictor) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-config-no-mtp.gguf").path();
    write_gguf(path, "qwen35", qwen35_metadata(4, 0));
    auto reader_r = GGUFReader::create(path);
    ASSERT_TRUE(reader_r.has_value());

    auto cfg_r = GgufConfigLoader::load(**reader_r);
    ASSERT_TRUE(cfg_r.has_value());
    const auto& cfg = *cfg_r;
    EXPECT_EQ(cfg.n_layers_, 24);
    EXPECT_EQ(cfg.nextn_predict_layers_, 0);
    ASSERT_EQ(cfg.layer_types_.size(), 24u);
    EXPECT_EQ(std::count(cfg.layer_types_.begin(), cfg.layer_types_.end(), LayerType::MtpPredictor), 0);
}

}  // namespace
}  // namespace ccinfer
