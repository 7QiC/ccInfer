#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

enum class DType : uint8_t {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
};

inline constexpr size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32:  return 4;
        case DType::Float16:  return 2;
        case DType::BFloat16: return 2;
        case DType::Int8:     return 1;
        case DType::Int4:     return 1;
    }
    return 0;
}

inline constexpr std::string_view dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32:  return "float32";
        case DType::Float16:  return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int8:     return "int8";
        case DType::Int4:     return "int4";
    }
    return "unknown";
}
