#include "checkpoint/huggingface/weight_source.h"

#include <string>
#include <utility>

namespace ccinfer {

SafetensorsWeightSource::SafetensorsWeightSource(std::shared_ptr<Reader> reader)
    : reader_(std::move(reader)) {}

Result<std::unique_ptr<SafetensorsWeightSource>> SafetensorsWeightSource::open(
    const std::string& path) {
    auto reader_r = Reader::create(path);
    if (!reader_r) return std::unexpected(reader_r.error());
    return std::unique_ptr<SafetensorsWeightSource>(
        new SafetensorsWeightSource(std::shared_ptr<Reader>(std::move(*reader_r))));
}

bool SafetensorsWeightSource::has(std::string_view name) const {
    return reader_->has(name);
}

Result<WeightSourceTensorInfo> SafetensorsWeightSource::info(std::string_view name) const {
    auto tensor = reader_->find_tensor(name);
    if (!tensor.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);

    WeightSourceTensorInfo out;
    out.logical_shape = tensor->shape_;
    out.storage_type = tensor->dtype_;
    out.offset = tensor->offset_;
    out.size_bytes = tensor->size_bytes_;
    return out;
}

Result<std::span<const uint8_t>> SafetensorsWeightSource::read(std::string_view name) const {
    auto tensor = reader_->find_tensor(name);
    if (!tensor.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
    return reader_->raw_tensor_data(*tensor);
}

}  // namespace ccinfer
