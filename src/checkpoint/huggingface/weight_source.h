#pragma once

#include <memory>
#include <string>

#include "base/error.h"
#include "checkpoint/huggingface/reader.h"
#include "model/weight_source.h"

namespace ccinfer {

class SafetensorsWeightSource final : public WeightSource {
public:
    explicit SafetensorsWeightSource(std::shared_ptr<Reader> reader);

    static Result<std::unique_ptr<SafetensorsWeightSource>> open(const std::string& path);

    bool has(std::string_view name) const override;
    Result<WeightSourceTensorInfo> info(std::string_view name) const override;
    Result<std::span<const uint8_t>> read(std::string_view name) const override;

private:
    std::shared_ptr<Reader> reader_;
};

}  // namespace ccinfer
