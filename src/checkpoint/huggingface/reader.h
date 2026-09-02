#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "base/error.h"
#include "facade/ops.h"

namespace ccinfer {

struct SafetensorsTensorInfo {
    ccop::DType dtype_ = ccop::DType::kUnknown;
    std::vector<int64_t> shape_;
    uint64_t offset_ = 0;
    uint64_t size_bytes_ = 0;
};

// Pure file-fact reader for single-file safetensors. It exposes only header
// metadata and raw bytes; no Backend allocation and no model semantics.
class Reader {
public:
    static constexpr size_t kHeaderSize = 8;

    static Result<std::unique_ptr<Reader>> create(const std::string& path);

    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    bool has(std::string_view name) const;
    std::optional<SafetensorsTensorInfo> find_tensor(std::string_view name) const;
    const std::vector<SafetensorsTensorInfo>& tensors() const noexcept;

    Result<std::span<const uint8_t>> raw_tensor_data(const SafetensorsTensorInfo& info) const;

private:
    Reader() = default;
    Result<void> parse();

    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    uint64_t data_start_ = 0;
    std::vector<SafetensorsTensorInfo> ordered_tensors_;
    std::unordered_map<std::string, SafetensorsTensorInfo> tensors_;
};

}  // namespace ccinfer
