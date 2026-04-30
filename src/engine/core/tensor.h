#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "common/dtype.h"

namespace ccinfer {
namespace engine {

template <int MaxRank = 5>
struct Tensor {
    void* data_ = nullptr;
    DType dtype_ = DType::kFloat32;
    int rank_ = 0;
    std::array<int64_t, MaxRank> shape_{};
    std::array<int64_t, MaxRank> stride_{};

    static Tensor make(void* data, DType dtype, std::initializer_list<int64_t> shape) {
        assert(static_cast<int>(shape.size()) <= MaxRank);
        Tensor t;
        t.data_ = data;
        t.dtype_ = dtype;
        t.rank_ = static_cast<int>(shape.size());
        int i = 0;
        for (auto s : shape) t.shape_[i++] = s;
        int64_t st = 1;
        for (int d = t.rank_ - 1; d >= 0; d--) {
            t.stride_[d] = st;
            st *= t.shape_[d];
        }
        return t;
    }

    [[nodiscard]] int64_t numel() const noexcept {
        int64_t n = 1;
        for (int i = 0; i < rank_; i++) n *= shape_[i];
        return n;
    }

    [[nodiscard]] size_t nbytes() const noexcept {
        return static_cast<size_t>(numel()) * dtype_size(dtype_);
    }

    [[nodiscard]] bool is_contiguous() const noexcept {
        int64_t expected = 1;
        for (int d = rank_ - 1; d >= 0; --d) {
            if (stride_[d] != expected) return false;
            expected *= shape_[d];
        }
        return true;
    }

    [[nodiscard]] Tensor slice(int dim, int64_t start, int64_t end) const {
        assert(dim >= 0 && dim < rank_);
        assert(start >= 0 && end >= start && end <= shape_[dim]);

        Tensor t = *this;
        t.shape_[dim] = end - start;
        t.data_ = static_cast<char*>(t.data_) + start * stride_[dim] * dtype_size(dtype_);
        return t;
    }

    [[nodiscard]] Tensor select(int dim, int64_t idx) const {
        assert(dim >= 0 && dim < rank_);
        assert(idx >= 0 && idx < shape_[dim]);

        Tensor t = *this;
        t.data_ = static_cast<char*>(t.data_) + idx * stride_[dim] * dtype_size(dtype_);
        for (int i = dim; i < t.rank_ - 1; i++) {
            t.shape_[i] = t.shape_[i + 1];
            t.stride_[i] = t.stride_[i + 1];
        }
        t.shape_[t.rank_ - 1] = 0;
        t.stride_[t.rank_ - 1] = 0;
        t.rank_--;
        return t;
    }
};

}  // namespace engine
}  // namespace ccinfer
