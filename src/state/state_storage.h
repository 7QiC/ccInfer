#pragma once

#include <memory>
#include <vector>

#include "backend/backend.h"
#include "base/error.h"
#include "config/model_config.h"
#include "runtime/tensor.h"

namespace ccinfer {

// Physical GDN state storage. One slot is a complete sequence state:
// for every GDN layer it contains the recurrent SSM matrix and the causal
// convolution history. StateStorage owns the device buffers and supports
// zero/copy at slot granularity. The model consumes this concrete type, just
// as it consumes BlockStorage directly (no extra abstract layer).
class StateStorage {
public:
    static Result<std::unique_ptr<StateStorage>> create(Backend& backend, const ModelConfig& config,
                                                        int max_slots);

    int num_slots() const;
    Tensor recurrent_state(int gdn_layer);
    Tensor conv_state(int gdn_layer);

    Result<void> zero_slot(int slot);
    Result<void> copy_slot(int src_slot, int dst_slot);

    int num_gdn_layers() const { return num_gdn_layers_; }

private:
    Backend* backend_ = nullptr;
    std::vector<Tensor> recurrent_layers_;
    std::vector<Tensor> conv_layers_;
    int num_gdn_layers_ = 0;
    int num_slots_ = 0;
};

}  // namespace ccinfer
