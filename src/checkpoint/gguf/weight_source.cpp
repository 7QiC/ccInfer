#include "checkpoint/gguf/weight_source.h"

#include <string>
#include <utility>

namespace ccinfer {

GGUFWeightSource::GGUFWeightSource(std::shared_ptr<GGUFReader> reader)
    : reader_(std::move(reader)) {}

Result<std::unique_ptr<GGUFWeightSource>> GGUFWeightSource::open(const std::string& path) {
    auto reader_r = GGUFReader::create(path);
    if (!reader_r) return std::unexpected(reader_r.error());
    return std::unique_ptr<GGUFWeightSource>(
        new GGUFWeightSource(std::shared_ptr<GGUFReader>(std::move(*reader_r))));
}

bool GGUFWeightSource::has(std::string_view name) const {
    return reader_->find_tensor(name).has_value();
}

Result<WeightSourceTensorInfo> GGUFWeightSource::info(std::string_view name) const {
    auto tensor = reader_->find_tensor(name);
    if (!tensor.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);

    WeightSourceTensorInfo out;
    out.logical_shape = tensor->dims;
    switch (tensor->type) {
        case GGUFTensorType::kQ8_0:
            out.storage_type = ccop::QuantType::kQ8_0;
            break;
        case GGUFTensorType::kF32:
            out.storage_type = ccop::DType::kFloat32;
            break;
        case GGUFTensorType::kF16:
            out.storage_type = ccop::DType::kFloat16;
            break;
        case GGUFTensorType::kBF16:
            out.storage_type = ccop::DType::kBFloat16;
            break;
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
    out.offset = tensor->offset;

    auto bytes = reader_->tensor_bytes(*tensor);
    if (!bytes) return std::unexpected(bytes.error());
    out.size_bytes = *bytes;
    return out;
}

Result<std::span<const uint8_t>> GGUFWeightSource::read(std::string_view name) const {
    auto tensor = reader_->find_tensor(name);
    if (!tensor.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
    return reader_->raw_tensor_data(*tensor);
}

}  // namespace ccinfer
