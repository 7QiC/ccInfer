#include "state/state_storage.h"

#include <cassert>
#include <utility>
#include <vector>

#include "runtime/precision.h"

namespace ccinfer {

namespace {

int count_gdn_layers(const ModelConfig& config) {
    int count = 0;
    for (std::size_t i = 0;
         i < config.layer_types_.size() && static_cast<int>(i) < config.n_layers_; ++i) {
        if (config.layer_types_[i] == LayerType::GatedDeltaNet) ++count;
    }
    return count;
}

std::size_t slot_bytes(const Tensor& layer_tensor, int num_slots) {
    assert(num_slots > 0);
    return layer_tensor.nbytes() / static_cast<std::size_t>(num_slots);
}

}  // namespace

Result<std::unique_ptr<StateStorage>> StateStorage::create(Backend& backend,
                                                           const ModelConfig& config,
                                                           int max_slots) {
    if (config.arch_ != ModelArch::Qwen3_5 || max_slots <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const int n_gdn = count_gdn_layers(config);
    if (n_gdn <= 0) return std::unexpected(ErrorCode::InvalidArgument);

    const int n_v = config.ssm_time_step_rank_;
    const int head_k = config.ssm_state_size_;
    const int head_v = config.ssm_inner_size_ / config.ssm_time_step_rank_;
    const int key_dim = config.ssm_group_count_ * head_k;
    const int value_dim = n_v * head_v;
    const int conv_dim = 2 * key_dim + value_dim;
    const int conv_state_len = config.ssm_conv_kernel_ - 1;

    auto storage = std::make_unique<StateStorage>();
    storage->backend_ = &backend;
    storage->num_gdn_layers_ = n_gdn;
    storage->num_slots_ = max_slots;

    storage->recurrent_layers_.reserve(static_cast<std::size_t>(n_gdn));
    storage->conv_layers_.reserve(static_cast<std::size_t>(n_gdn));

    for (int layer = 0; layer < n_gdn; ++layer) {
        auto rec_r =
            Tensor::empty(backend, ccop::DType::kFloat32, {max_slots, n_v, head_k, head_v});
        if (!rec_r) return std::unexpected(rec_r.error());
        auto conv_r =
            Tensor::empty(backend, ccop::DType::kBFloat16, {max_slots, conv_dim, conv_state_len});
        if (!conv_r) return std::unexpected(conv_r.error());

        if (auto r = backend.memset(rec_r->data(), 0, rec_r->nbytes()); !r) {
            return std::unexpected(r.error());
        }
        if (auto r = backend.memset(conv_r->data(), 0, conv_r->nbytes()); !r) {
            return std::unexpected(r.error());
        }

        storage->recurrent_layers_.push_back(std::move(*rec_r));
        storage->conv_layers_.push_back(std::move(*conv_r));
    }
    if (auto r = backend.synchronize(); !r) return std::unexpected(r.error());
    return storage;
}

int StateStorage::num_slots() const { return num_slots_; }

Tensor StateStorage::recurrent_state(int gdn_layer) {
    assert(gdn_layer >= 0 && gdn_layer < num_gdn_layers_);
    return recurrent_layers_[static_cast<std::size_t>(gdn_layer)];
}

Tensor StateStorage::conv_state(int gdn_layer) {
    assert(gdn_layer >= 0 && gdn_layer < num_gdn_layers_);
    return conv_layers_[static_cast<std::size_t>(gdn_layer)];
}

Result<void> StateStorage::zero_slot(int slot) {
    if (slot < 0 || slot >= num_slots_) return std::unexpected(ErrorCode::InvalidArgument);
    assert(backend_ != nullptr);
    for (auto& layer : recurrent_layers_) {
        const std::size_t bytes = slot_bytes(layer, num_slots_);
        auto* data = static_cast<char*>(layer.data()) + static_cast<std::size_t>(slot) * bytes;
        if (auto r = backend_->memset(data, 0, bytes); !r) return r;
    }
    for (auto& layer : conv_layers_) {
        const std::size_t bytes = slot_bytes(layer, num_slots_);
        auto* data = static_cast<char*>(layer.data()) + static_cast<std::size_t>(slot) * bytes;
        if (auto r = backend_->memset(data, 0, bytes); !r) return r;
    }
    return {};
}

Result<void> StateStorage::copy_slot(int src_slot, int dst_slot) {
    if (src_slot < 0 || src_slot >= num_slots_ || dst_slot < 0 || dst_slot >= num_slots_) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    assert(backend_ != nullptr);
    for (auto& layer : recurrent_layers_) {
        const std::size_t bytes = slot_bytes(layer, num_slots_);
        auto* src = static_cast<char*>(layer.data()) + static_cast<std::size_t>(src_slot) * bytes;
        auto* dst = static_cast<char*>(layer.data()) + static_cast<std::size_t>(dst_slot) * bytes;
        if (auto r = backend_->memcpy_d2d(dst, src, bytes); !r) return r;
    }
    for (auto& layer : conv_layers_) {
        const std::size_t bytes = slot_bytes(layer, num_slots_);
        auto* src = static_cast<char*>(layer.data()) + static_cast<std::size_t>(src_slot) * bytes;
        auto* dst = static_cast<char*>(layer.data()) + static_cast<std::size_t>(dst_slot) * bytes;
        if (auto r = backend_->memcpy_d2d(dst, src, bytes); !r) return r;
    }
    return {};
}

}  // namespace ccinfer
