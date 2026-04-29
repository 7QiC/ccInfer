#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include "dtype.h"

namespace ccinfer {

template<int MaxRank = 5>
struct Tensor {
    void*   data = nullptr;
    DType   dtype = DType::kFloat32;
    int     rank = 0;
    std::array<int64_t, MaxRank> shape{};
    std::array<int64_t, MaxRank> stride{};

    static Tensor make(void* data, DType dtype, std::initializer_list<int64_t> shape) {
        Tensor t;
        t.data = data;
        t.dtype = dtype;
        t.rank = static_cast<int>(shape.size());
        int i = 0;
        for (auto s : shape) t.shape[i++] = s;
        int64_t st = 1;
        for (int d = t.rank - 1; d >= 0; d--) {
            t.stride[d] = st;
            st *= t.shape[d];
        }
        return t;
    }

    int64_t numel() const noexcept {
        int64_t n = 1;
        for (int i = 0; i < rank; i++) n *= shape[i];
        return n;
    }

    size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * dtype_size(dtype);
    }

    Tensor slice(int dim, int64_t start, int64_t end) const {
        Tensor t = *this;
        t.shape[dim] = end - start;
        t.data = static_cast<char*>(t.data) + start * stride[dim] * dtype_size(dtype);
        return t;
    }

    Tensor select(int dim, int64_t idx) const {
        Tensor t = *this;
        t.data = static_cast<char*>(t.data) + idx * stride[dim] * dtype_size(dtype);
        for (int i = dim; i < t.rank - 1; i++) {
            t.shape[i] = t.shape[i + 1];
            t.stride[i] = t.stride[i + 1];
        }
        t.rank--;
        return t;
    }
};

} // namespace ccinfer
