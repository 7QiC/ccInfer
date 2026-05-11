#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ccinfer {
namespace engine {

enum class DType : uint8_t {
    kUnknown,
    kFloat32,
    kFloat16,
    kBFloat16,
    kInt8,
};

inline constexpr size_t dtype_size(DType dt) noexcept {
    switch (dt) {
        case DType::kFloat32:
            return 4;
        case DType::kFloat16:
            return 2;
        case DType::kBFloat16:
            return 2;
        case DType::kInt8:
            return 1;
    }
    return 0;
}

inline constexpr std::string_view dtype_name(DType dt) noexcept {
    switch (dt) {
        case DType::kFloat32:
            return "float32";
        case DType::kFloat16:
            return "float16";
        case DType::kBFloat16:
            return "bfloat16";
        case DType::kInt8:
            return "int8";
    }
    return "unknown";
}

}  // namespace engine
}  // namespace ccinfer
