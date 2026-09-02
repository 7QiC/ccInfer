#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "base/error.h"

namespace ccinfer {

enum class GGUFValueType : uint32_t {
    kU8 = 0,
    kI8 = 1,
    kU16 = 2,
    kI16 = 3,
    kU32 = 4,
    kI32 = 5,
    kF32 = 6,
    kBool = 7,
    kString = 8,
    kArray = 9,
    kU64 = 10,
    kI64 = 11,
    kF64 = 12,
};

enum class GGUFTensorType : uint32_t {
    kF32 = 0,
    kF16 = 1,
    kQ4_0 = 2,
    kQ4_1 = 3,
    kQ5_0 = 6,
    kQ5_1 = 7,
    kQ8_0 = 8,
    kQ8_1 = 9,
    kQ2_K = 10,
    kQ3_K = 11,
    kQ4_K = 12,
    kQ5_K = 13,
    kQ6_K = 14,
    kQ8_K = 15,
    kIQ2_XXS = 16,
    kIQ2_XS = 17,
    kIQ3_XXS = 18,
    kIQ1_S = 19,
    kIQ4_NL = 20,
    kIQ3_S = 21,
    kIQ2_S = 22,
    kIQ4_XS = 23,
    kI8 = 24,
    kI16 = 25,
    kI32 = 26,
    kI64 = 27,
    kF64 = 28,
    kIQ1_M = 29,
    kBF16 = 30,
};

struct GGUFMetadataValue {
    GGUFValueType type = GGUFValueType::kU8;
    std::variant<std::monostate, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                 uint64_t, int64_t, float, double, bool, std::string,
                 std::vector<GGUFMetadataValue>> value;

    const std::string& as_string() const;
    std::optional<uint64_t> as_u64() const;
    std::optional<int64_t> as_i64() const;
    std::optional<double> as_f64() const;
    std::optional<float> as_f32() const;
    const std::vector<GGUFMetadataValue>* as_array() const;
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    GGUFTensorType type = GGUFTensorType::kF32;
    uint64_t offset = 0;
};

class GGUFReader {
public:
    static Result<std::unique_ptr<GGUFReader>> create(const std::string& path);

    ~GGUFReader();
    GGUFReader(const GGUFReader&) = delete;
    GGUFReader& operator=(const GGUFReader&) = delete;

    uint32_t version() const noexcept { return version_; }
    uint64_t tensor_count() const noexcept { return tensor_infos_.size(); }
    uint64_t metadata_count() const noexcept { return metadata_.size(); }

    const std::vector<std::pair<std::string, GGUFMetadataValue>>& metadata() const noexcept {
        return metadata_;
    }
    std::optional<GGUFMetadataValue> metadata(std::string_view key) const;

    const std::vector<GGUFTensorInfo>& tensors() const noexcept { return tensor_infos_; }
    std::optional<GGUFTensorInfo> find_tensor(std::string_view name) const;

    Result<uint64_t> tensor_bytes(const GGUFTensorInfo& info) const;
    Result<std::span<const uint8_t>> raw_tensor_data(const GGUFTensorInfo& info) const;

    // Convenience for M0 inventory use; does not interpret qwen35 semantics.
    bool has_nextn_tensors() const noexcept;

private:
    GGUFReader() = default;
    Result<void> parse(const uint8_t* data, size_t size);

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    uint64_t data_start_ = 0;
    uint32_t version_ = 0;
    std::vector<std::pair<std::string, GGUFMetadataValue>> metadata_;
    std::vector<GGUFTensorInfo> tensor_infos_;
};

}  // namespace ccinfer
