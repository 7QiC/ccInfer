#pragma once

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "backend/buffer.h"
#include "ccop/quant.h"
#include "runtime/tensor.h"

namespace ccinfer {

enum class WeightKind : uint8_t { kDense, kQuantized };

struct DenseWeight {
    Tensor tensor;
};

struct QuantizedWeight {
    std::shared_ptr<Buffer> storage;
    std::vector<int64_t> logical_shape;
    ccop::QuantType quant_type = ccop::QuantType::kQ8_0;
};

using Weight = std::variant<DenseWeight, QuantizedWeight>;

WeightKind weight_kind(const Weight& weight) noexcept;

std::vector<int64_t> weight_logical_shape(const Weight& weight);

uint64_t weight_byte_size(const Weight& weight) noexcept;

}  // namespace ccinfer
