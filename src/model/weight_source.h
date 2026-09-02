#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

#include "base/error.h"
#include "facade/ops.h"
#include "ccop/quant.h"

namespace ccinfer {

// Storage representation of a tensor: either a dense ccop dtype or a quantized
// block format. This deliberately keeps Q8_0 from being treated as a dense
// dtype.
using TensorStorageType = std::variant<ccop::DType, ccop::QuantType>;

struct WeightSourceTensorInfo {
    std::vector<int64_t> logical_shape;
    TensorStorageType storage_type;
    uint64_t offset = 0;
    uint64_t size_bytes = 0;
};

// Format-agnostic tensor source. Implementations expose only existence, shape,
// storage kind, and raw bytes; they do not allocate device memory or interpret
// Qwen tensor names.
class WeightSource {
public:
    virtual ~WeightSource() = default;

    virtual bool has(std::string_view name) const = 0;
    virtual Result<WeightSourceTensorInfo> info(std::string_view name) const = 0;
    virtual Result<std::span<const uint8_t>> read(std::string_view name) const = 0;
};

}  // namespace ccinfer
