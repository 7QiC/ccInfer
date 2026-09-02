#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "checkpoint/gguf/weight_source.h"
#include "checkpoint/huggingface/weight_source.h"
#include "model/weight_source.h"
#include "ccop/quant.h"

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

TEST(WeightSourceTest, GgufDenseAndQuantized) {
    const std::string path = ScopedTempFile("ccinfer-test-weight-source.gguf").path();
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    f.write("GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 2);  // tensors
    write_u64(f, 0);  // metadata

    // Tensor 1: Q8_0 [2, 32]
    write_string(f, "q8_tensor");
    write_u32(f, 2);
    write_u64(f, 2);
    write_u64(f, 32);
    write_u32(f, 8);   // Q8_0
    write_u64(f, 0);

    // Tensor 2: F32 [3]
    write_string(f, "f32_tensor");
    write_u32(f, 1);
    write_u64(f, 3);
    write_u32(f, 0);   // F32
    write_u64(f, 68);

    std::vector<uint8_t> data(68 + 12);
    for (uint8_t& b : data) b = 0x42;
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    f.close();

    auto source_r = GGUFWeightSource::open(path);
    ASSERT_TRUE(source_r.has_value());
    auto& source = **source_r;

    EXPECT_TRUE(source.has("q8_tensor"));
    EXPECT_TRUE(source.has("f32_tensor"));
    EXPECT_FALSE(source.has("missing"));

    const auto q8_info = source.info("q8_tensor");
    ASSERT_TRUE(q8_info.has_value());
    EXPECT_EQ(q8_info->logical_shape, (std::vector<int64_t>{2, 32}));
    ASSERT_TRUE(std::holds_alternative<ccop::QuantType>(q8_info->storage_type));
    EXPECT_EQ(std::get<ccop::QuantType>(q8_info->storage_type), ccop::QuantType::kQ8_0);
    const std::vector<int64_t> q8_shape{2, 32};
    EXPECT_EQ(q8_info->size_bytes, *ccop::q8_0_storage_bytes(q8_shape));

    const auto f32_info = source.info("f32_tensor");
    ASSERT_TRUE(f32_info.has_value());
    ASSERT_TRUE(std::holds_alternative<ccop::DType>(f32_info->storage_type));
    EXPECT_EQ(std::get<ccop::DType>(f32_info->storage_type), ccop::DType::kFloat32);
    EXPECT_EQ(f32_info->size_bytes, 12u);

    const auto q8_raw = source.read("q8_tensor");
    ASSERT_TRUE(q8_raw.has_value());
    EXPECT_EQ(q8_raw->size(), 68u);
    for (uint8_t b : *q8_raw) EXPECT_EQ(b, 0x42);

    EXPECT_FALSE(source.info("missing").has_value());
    EXPECT_FALSE(source.read("missing").has_value());
}

TEST(WeightSourceTest, SafetensorsDense) {
    const std::string path = ScopedTempFile("ccinfer-test-weight-source.safetensors").path();
    const std::string header = R"({
        "bf16_tensor": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "f32_tensor": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]}
    })";

    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    const uint64_t header_len = header.size();
    f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::vector<uint8_t> data(28, 0x33);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    f.close();

    auto source_r = SafetensorsWeightSource::open(path);
    ASSERT_TRUE(source_r.has_value());
    auto& source = **source_r;

    EXPECT_TRUE(source.has("bf16_tensor"));
    EXPECT_TRUE(source.has("f32_tensor"));

    const auto bf16_info = source.info("bf16_tensor");
    ASSERT_TRUE(bf16_info.has_value());
    EXPECT_EQ(bf16_info->logical_shape, (std::vector<int64_t>{2, 3}));
    ASSERT_TRUE(std::holds_alternative<ccop::DType>(bf16_info->storage_type));
    EXPECT_EQ(std::get<ccop::DType>(bf16_info->storage_type), ccop::DType::kBFloat16);
    EXPECT_EQ(bf16_info->size_bytes, 12u);

    const auto f32_info = source.info("f32_tensor");
    ASSERT_TRUE(f32_info.has_value());
    EXPECT_EQ(std::get<ccop::DType>(f32_info->storage_type), ccop::DType::kFloat32);
    EXPECT_EQ(f32_info->size_bytes, 16u);

    const auto raw = source.read("bf16_tensor");
    ASSERT_TRUE(raw.has_value());
    EXPECT_EQ(raw->size(), 12u);
    for (uint8_t b : *raw) EXPECT_EQ(b, 0x33);
}

}  // namespace
}  // namespace ccinfer
