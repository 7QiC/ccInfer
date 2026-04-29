#pragma once

#include <array>
#include <cassert>
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
        assert(static_cast<int>(shape.size()) <= MaxRank);
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

    [[nodiscard]] int64_t numel() const noexcept {
        int64_t n = 1;
        for (int i = 0; i < rank; i++) n *= shape[i];
        return n;
    }

    [[nodiscard]] size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * dtype_size(dtype);
    }

    [[nodiscard]] bool is_contiguous() const noexcept {
        int64_t expected = 1;
        for (int d = rank - 1; d >= 0; --d) {
            if (stride[d] != expected) {
                return false;
            }
            expected *= shape[d];
        }
        return true;
    }

    [[nodiscard]] Tensor slice(int dim, int64_t start, int64_t end) const {
        assert(dim >= 0 && dim < rank);
        assert(start >= 0 && end >= start && end <= shape[dim]);

        Tensor t = *this;
        t.shape[dim] = end - start;
        t.data = static_cast<char*>(t.data) + start * stride[dim] * dtype_size(dtype);
        return t;
    }

    [[nodiscard]] Tensor select(int dim, int64_t idx) const {
        assert(dim >= 0 && dim < rank);
        assert(idx >= 0 && idx < shape[dim]);

        Tensor t = *this;
        t.data = static_cast<char*>(t.data) + idx * stride[dim] * dtype_size(dtype);
        for (int i = dim; i < t.rank - 1; i++) {
            t.shape[i] = t.shape[i + 1];
            t.stride[i] = t.stride[i + 1];
        }
        t.shape[t.rank - 1] = 0;
        t.stride[t.rank - 1] = 0;
        t.rank--;
        return t;
    }
};

} // namespace ccinfer
