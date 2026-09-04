#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

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

void write_metadata(std::ofstream& f, const std::string& key, uint32_t type,
                    const std::vector<uint8_t>& value) {
    write_string(f, key);
    write_u32(f, type);
    f.write(reinterpret_cast<const char*>(value.data()), static_cast<std::streamsize>(value.size()));
}

std::vector<uint8_t> scalar_u32(uint32_t v) {
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

std::vector<uint8_t> array_u32(uint32_t elem_type, const std::vector<uint32_t>& values) {
    std::vector<uint8_t> bytes;
    const uint8_t* ep = reinterpret_cast<const uint8_t*>(&elem_type);
    bytes.insert(bytes.end(), ep, ep + sizeof(elem_type));
    const uint64_t count = values.size();
    const uint8_t* cp = reinterpret_cast<const uint8_t*>(&count);
    bytes.insert(bytes.end(), cp, cp + sizeof(count));
    for (uint32_t v : values) {
        const uint8_t* vp = reinterpret_cast<const uint8_t*>(&v);
        bytes.insert(bytes.end(), vp, vp + sizeof(v));
    }
    return bytes;
}

TEST(GGUFReaderTest, SyntheticMetadataAndTensors) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-synthetic.gguf").path();
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    f.write("GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 1);  // n_tensors
    write_u64(f, 3);  // n_metadata

    write_metadata(f, "general.architecture", 8, scalar_string("qwen35"));
    write_metadata(f, "qwen35.block_count", 4, scalar_u32(25));
    write_metadata(f, "qwen35.rope.dimension_sections", 9,
                   array_u32(4, {11, 11, 10, 0}));

    write_string(f, "token_embd.weight");
    write_u32(f, 2);
    write_u64(f, 32);  // GGUF ne[0] is the block-aligned innermost dim
    write_u64(f, 2);
    write_u32(f, 8);  // Q8_0
    write_u64(f, 0);

    const std::vector<uint8_t> tensor_data(2 * 34, 0xAB);
    f.write(reinterpret_cast<const char*>(tensor_data.data()),
            static_cast<std::streamsize>(tensor_data.size()));
    f.close();

    auto reader_r = GGUFReader::create(path);
    ASSERT_TRUE(reader_r.has_value());
    auto& reader = **reader_r;

    EXPECT_EQ(reader.version(), 3u);
    EXPECT_EQ(reader.tensor_count(), 1u);
    EXPECT_EQ(reader.metadata_count(), 3u);
    EXPECT_EQ(reader.metadata("general.architecture")->as_string(), "qwen35");
    EXPECT_EQ(reader.metadata("qwen35.block_count")->as_u64().value(), 25u);

    auto rope = reader.metadata("qwen35.rope.dimension_sections");
    ASSERT_TRUE(rope.has_value());
    auto* arr = rope->as_array();
    ASSERT_NE(arr, nullptr);
    ASSERT_EQ(arr->size(), 4u);
    EXPECT_EQ((*arr)[0].as_u64().value(), 11u);
    EXPECT_EQ((*arr)[3].as_u64().value(), 0u);

    auto tensor = reader.find_tensor("token_embd.weight");
    ASSERT_TRUE(tensor.has_value());
    ASSERT_EQ(tensor->dims.size(), 2u);
    EXPECT_EQ(tensor->dims[0], 32);
    EXPECT_EQ(tensor->dims[1], 2);
    auto bytes_r = reader.tensor_bytes(*tensor);
    ASSERT_TRUE(bytes_r.has_value());
    EXPECT_EQ(*bytes_r, 68u);
    auto raw_r = reader.raw_tensor_data(*tensor);
    ASSERT_TRUE(raw_r.has_value());
    ASSERT_EQ(raw_r->size(), 68u);
    EXPECT_FALSE(reader.has_nextn_tensors());
}

TEST(GGUFReaderTest, RejectsBadMagic) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-badmagic.gguf").path();
    std::ofstream f(path, std::ios::binary);
    f.write("NOPE", 4);
    write_u32(f, 3);
    write_u64(f, 0);
    write_u64(f, 0);
    f.close();

    auto r = GGUFReader::create(path);
    EXPECT_FALSE(r.has_value());
}

TEST(GGUFReaderTest, RejectsUnsupportedVersion) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-version.gguf").path();
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    write_u32(f, 4);
    write_u64(f, 0);
    write_u64(f, 0);
    f.close();

    auto r = GGUFReader::create(path);
    EXPECT_FALSE(r.has_value());
}

TEST(GGUFReaderTest, RejectsUnsupportedTensorType) {
    const std::string path = ScopedTempFile("ccinfer-test-gguf-badtensor.gguf").path();
    std::ofstream f(path, std::ios::binary);
    f.write("GGUF", 4);
    write_u32(f, 3);
    write_u64(f, 1);
    write_u64(f, 0);

    write_string(f, "bad.tensor");
    write_u32(f, 2);
    write_u64(f, 2);
    write_u64(f, 32);
    write_u32(f, 2);  // Q4_0 unsupported for loading
    write_u64(f, 0);
    f.close();

    auto r = GGUFReader::create(path);
    EXPECT_FALSE(r.has_value());
}

#ifndef GGUF_ARTIFACT_PATH
#define GGUF_ARTIFACT_PATH "models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf"
#endif

TEST(GGUFReaderTest, RealArtifactSmoke) {
    const std::string path = GGUF_ARTIFACT_PATH;
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "GGUF artifact not present";

    auto r = GGUFReader::create(path);
    ASSERT_TRUE(r.has_value());
    auto& reader = **r;

    EXPECT_EQ(reader.tensor_count(), 335u);
    EXPECT_EQ(reader.metadata_count(), 48u);
    EXPECT_EQ(reader.metadata("general.architecture")->as_string(), "qwen35");
    EXPECT_TRUE(reader.has_nextn_tensors());
}

}  // namespace
}  // namespace ccinfer
