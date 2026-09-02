#include "model/weight.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>

namespace ccinfer {

WeightKind weight_kind(const Weight& weight) noexcept {
    return std::holds_alternative<DenseWeight>(weight) ? WeightKind::kDense : WeightKind::kQuantized;
}

std::vector<int64_t> weight_logical_shape(const Weight& weight) {
    return std::visit(
        [](const auto& w) -> std::vector<int64_t> {
            using T = std::decay_t<decltype(w)>;
            if constexpr (std::is_same_v<T, DenseWeight>) {
                std::vector<int64_t> shape;
                shape.reserve(static_cast<std::size_t>(w.tensor.rank()));
                const auto& src = w.tensor.shape();
                for (int i = 0; i < w.tensor.rank(); ++i) {
                    shape.push_back(src[static_cast<std::size_t>(i)]);
                }
                return shape;
            } else {
                return w.logical_shape;
            }
        },
        weight);
}

uint64_t weight_byte_size(const Weight& weight) noexcept {
    return std::visit(
        [](const auto& w) -> uint64_t {
            using T = std::decay_t<decltype(w)>;
            if constexpr (std::is_same_v<T, DenseWeight>) {
                return static_cast<uint64_t>(w.tensor.nbytes());
            } else {
                if (w.quant_type != ccop::QuantType::kQ8_0) return 0;
                auto bytes = ccop::q8_0_storage_bytes(w.logical_shape);
                return bytes ? static_cast<uint64_t>(*bytes) : 0;
            }
        },
        weight);
}

}  // namespace ccinfer
