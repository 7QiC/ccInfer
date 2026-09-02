#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "checkpoint/huggingface/reader.h"

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

TEST(ReaderTest, SyntheticHeaderAndRawBytes) {
    const std::string path = ScopedTempFile("ccinfer-test-safetensors.safetensors").path();
    const std::string header = R"({
        "__metadata__": {"format": "pt"},
        "tensor_bf16": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        "tensor_f32": {"dtype": "F32", "shape": [4], "data_offsets": [12, 28]}
    })";

    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());
    const uint64_t header_len = header.size();
    f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    f.write(header.data(), static_cast<std::streamsize>(header.size()));

    std::vector<uint8_t> data(28, 0x5A);
    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    f.close();

    auto reader_r = Reader::create(path);
    ASSERT_TRUE(reader_r.has_value());
    auto& reader = **reader_r;

    EXPECT_TRUE(reader.has("tensor_bf16"));
    EXPECT_TRUE(reader.has("tensor_f32"));
    EXPECT_FALSE(reader.has("missing"));

    auto bf16 = reader.find_tensor("tensor_bf16");
    ASSERT_TRUE(bf16.has_value());
    EXPECT_EQ(bf16->dtype_, ccop::DType::kBFloat16);
    EXPECT_EQ(bf16->shape_, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(bf16->size_bytes_, 12u);

    auto f32 = reader.find_tensor("tensor_f32");
    ASSERT_TRUE(f32.has_value());
    EXPECT_EQ(f32->dtype_, ccop::DType::kFloat32);
    EXPECT_EQ(f32->offset_, 12u);
    EXPECT_EQ(f32->size_bytes_, 16u);

    auto raw = reader.raw_tensor_data(*bf16);
    ASSERT_TRUE(raw.has_value());
    EXPECT_EQ(raw->size(), 12u);
    for (uint8_t b : *raw) EXPECT_EQ(b, 0x5A);
}

TEST(ReaderTest, RejectsTruncatedHeader) {
    const std::string path = ScopedTempFile("ccinfer-test-safetensors-truncated.safetensors").path();
    std::ofstream f(path, std::ios::binary);
    const uint64_t header_len = 100;
    f.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    f.write("short", 5);
    f.close();

    auto r = Reader::create(path);
    EXPECT_FALSE(r.has_value());
}

}  // namespace
}  // namespace ccinfer
