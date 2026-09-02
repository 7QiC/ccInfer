#pragma once

#include <memory>
#include <string>

#include "base/error.h"
#include "checkpoint/gguf/reader.h"
#include "model/weight_source.h"

namespace ccinfer {

class GGUFWeightSource final : public WeightSource {
public:
    explicit GGUFWeightSource(std::shared_ptr<GGUFReader> reader);

    static Result<std::unique_ptr<GGUFWeightSource>> open(const std::string& path);

    bool has(std::string_view name) const override;
    Result<WeightSourceTensorInfo> info(std::string_view name) const override;
    Result<std::span<const uint8_t>> read(std::string_view name) const override;

private:
    std::shared_ptr<GGUFReader> reader_;
};

}  // namespace ccinfer
