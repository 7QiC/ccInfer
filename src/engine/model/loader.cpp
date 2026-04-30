#include "engine/model/loader.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#include "engine/backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

namespace {

constexpr size_t kSafetensorsHeaderSize = 8;

Result<DType> parse_dtype(const std::string& dt) {
    if (dt == "BF16") {
        return DType::kBFloat16;
    }
    if (dt == "F16") {
        return DType::kFloat16;
    }
    if (dt == "F32") {
        return DType::kFloat32;
    }
    return std::unexpected(ErrorCode::ModelLoadFailed);
}

Result<int64_t> checked_numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;

    for (int64_t s : shape) {
        if (s < 0) {
            return std::unexpected(ErrorCode::ModelShapeMismatch);
        }

        if (s == 0) {
            return int64_t{0};
        }

        if (n > std::numeric_limits<int64_t>::max() / s) {
            return std::unexpected(ErrorCode::ModelShapeMismatch);
        }

        n *= s;
    }

    return n;
}

Result<void> check_shape_equal(const std::vector<int64_t>& actual,
                               const std::vector<int64_t>& expected) {
    if (actual != expected) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    return {};
}

}  // namespace

WeightLoader::WeightLoader(std::string path) : path_(std::move(path)) {}

WeightLoader::~WeightLoader() { cleanup(); }

WeightLoader::WeightLoader(WeightLoader&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      header_len_(std::exchange(other.header_len_, 0)),
      tensors_(std::move(other.tensors_)) {}

WeightLoader& WeightLoader::operator=(WeightLoader&& other) noexcept {
    if (this != &other) {
        cleanup();

        path_ = std::move(other.path_);
        fd_ = std::exchange(other.fd_, -1);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        header_len_ = std::exchange(other.header_len_, 0);
        tensors_ = std::move(other.tensors_);
    }

    return *this;
}

void WeightLoader::cleanup() noexcept {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    size_ = 0;
    header_len_ = 0;
    tensors_.clear();
}

Result<WeightLoader> WeightLoader::create(const std::string& path) {
    WeightLoader loader(path);

    loader.fd_ = open(path.c_str(), O_RDONLY);
    if (loader.fd_ < 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    struct stat st {};
    if (fstat(loader.fd_, &st) < 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    if (st.st_size <= 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    loader.size_ = static_cast<size_t>(st.st_size);

    void* mapped = mmap(nullptr, loader.size_, PROT_READ, MAP_PRIVATE, loader.fd_, 0);

    if (mapped == MAP_FAILED) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    loader.data_ = static_cast<uint8_t*>(mapped);

    auto parsed = loader.parse();
    if (!parsed) {
        return std::unexpected(parsed.error());
    }

    return std::move(loader);
}

Result<void> WeightLoader::parse() {
    if (data_ == nullptr || size_ < kSafetensorsHeaderSize) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    uint64_t header_len = 0;
    std::memcpy(&header_len, data_, sizeof(uint64_t));

    if (header_len == 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    if (kSafetensorsHeaderSize + header_len > size_) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    header_len_ = header_len;

    const auto* header_begin = reinterpret_cast<const char*>(data_ + kSafetensorsHeaderSize);

    std::string header_str(header_begin, static_cast<size_t>(header_len_));

    auto j = nlohmann::json::parse(header_str, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const uint64_t data_start = kSafetensorsHeaderSize + header_len_;

    for (auto& [name, info] : j.items()) {
        // safetensors commonly contains this optional entry.
        if (name == "__metadata__") {
            continue;
        }

        if (!info.is_object()) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        if (!info.contains("dtype") || !info.contains("shape") || !info.contains("data_offsets")) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        if (!info["dtype"].is_string() || !info["shape"].is_array() ||
            !info["data_offsets"].is_array() || info["data_offsets"].size() != 2) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        TensorInfo t;

        auto dtype_result = parse_dtype(info["dtype"].get<std::string>());
        if (!dtype_result) {
            return std::unexpected(dtype_result.error());
        }
        t.dtype_ = *dtype_result;

        for (const auto& s : info["shape"]) {
            if (!s.is_number_integer()) {
                return std::unexpected(ErrorCode::ModelLoadFailed);
            }
            t.shape_.push_back(s.get<int64_t>());
        }

        const auto numel_result = checked_numel(t.shape_);
        if (!numel_result) {
            return std::unexpected(numel_result.error());
        }

        const uint64_t begin = info["data_offsets"][0].get<uint64_t>();
        const uint64_t end = info["data_offsets"][1].get<uint64_t>();

        if (end < begin) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        t.offset_ = begin;
        t.size_bytes_ = end - begin;

        if (data_start + t.offset_ > size_) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        if (t.size_bytes_ > size_ - data_start - t.offset_) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        const int64_t numel = *numel_result;

        // Phase 2 BF16-only loader.
        // We still parse dtype info for diagnostics, but later load() rejects non-BF16.
        if (t.dtype_ == DType::kBFloat16) {
            const uint64_t expected_bytes = static_cast<uint64_t>(numel) * sizeof(__nv_bfloat16);

            if (t.size_bytes_ != expected_bytes) {
                return std::unexpected(ErrorCode::ModelShapeMismatch);
            }
        }

        tensors_.emplace(name, std::move(t));
    }

    if (tensors_.empty()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    return {};
}

Result<DeviceBuffer<__nv_bfloat16>> WeightLoader::load(
    const std::string& name, const std::vector<int64_t>& expected_shape) const {
    if (data_ == nullptr || size_ == 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const TensorInfo& info = it->second;

    auto shape_ok = check_shape_equal(info.shape_, expected_shape);
    if (!shape_ok) {
        return std::unexpected(shape_ok.error());
    }

    // Critical: this loader returns DeviceBuffer<__nv_bfloat16>.
    // So the source tensor must be BF16.
    if (info.dtype_ != DType::kBFloat16) {
        // 如果你的 ErrorCode 里还没有 ModelUnsupportedDType，
        // 建议加一个。暂时可以用 ModelLoadFailed 替代。
        return std::unexpected(ErrorCode::ModelUnsupportedDType);
    }

    auto numel_result = checked_numel(info.shape_);
    if (!numel_result) {
        return std::unexpected(numel_result.error());
    }

    const int64_t numel = *numel_result;
    const size_t numel_size = static_cast<size_t>(numel);

    const uint64_t expected_bytes = static_cast<uint64_t>(numel) * sizeof(__nv_bfloat16);

    if (info.size_bytes_ != expected_bytes) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const uint64_t data_start = kSafetensorsHeaderSize + header_len_;

    if (data_start + info.offset_ > size_) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    if (info.size_bytes_ > size_ - data_start - info.offset_) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const uint8_t* tensor_data = data_ + data_start + info.offset_;

    DeviceBuffer<__nv_bfloat16> buf(numel_size);

    auto copy_result = cuda_check(cudaMemcpy(
        buf.get(), tensor_data, static_cast<size_t>(info.size_bytes_), cudaMemcpyHostToDevice));

    if (!copy_result) {
        return std::unexpected(copy_result.error());
    }

    return buf;
}

Result<ModelWeights> WeightLoader::load_all(const ModelConfig& cfg) const {
    ModelWeights w;

    const int d = cfg.d_model_;
    const int nq = cfg.n_q_heads_;
    const int nkv = cfg.n_kv_heads_;
    const int hd = cfg.head_dim_;
    const int n_layers = cfg.n_layers_;
    const int d_ff = cfg.d_ff_;
    const int vocab = cfg.vocab_size_;

    if (d <= 0 || nq <= 0 || nkv <= 0 || hd <= 0 || n_layers <= 0 || d_ff <= 0 || vocab <= 0) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    if (nq * hd != d) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    if (nq % nkv != 0) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    auto embed = load("model.embed_tokens.weight", {vocab, d});
    if (!embed) {
        return std::unexpected(embed.error());
    }
    w.embed_ = std::move(*embed);

    // Phase 2: first try lm_head.weight.
    // Some tied-weight models may not have it; for now we return error.
    auto lm_head = load("lm_head.weight", {vocab, d});
    if (!lm_head) {
        return std::unexpected(lm_head.error());
    }
    w.lm_head_ = std::move(*lm_head);

    auto rms_final = load("model.norm.weight", {d});
    if (!rms_final) {
        return std::unexpected(rms_final.error());
    }
    w.rms_final_ = std::move(*rms_final);

    w.layers_.reserve(static_cast<size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i);

        LayerWeights lw;

        auto o = load(p + ".self_attn.o_proj.weight", {d, nq * hd});
        if (!o) {
            return std::unexpected(o.error());
        }
        lw.o_ = std::move(*o);

        auto gate = load(p + ".mlp.gate_proj.weight", {d_ff, d});
        if (!gate) {
            return std::unexpected(gate.error());
        }
        lw.gate_ = std::move(*gate);

        auto up = load(p + ".mlp.up_proj.weight", {d_ff, d});
        if (!up) {
            return std::unexpected(up.error());
        }
        lw.up_ = std::move(*up);

        auto down = load(p + ".mlp.down_proj.weight", {d, d_ff});
        if (!down) {
            return std::unexpected(down.error());
        }
        lw.down_ = std::move(*down);

        auto rms_attn = load(p + ".input_layernorm.weight", {d});
        if (!rms_attn) {
            return std::unexpected(rms_attn.error());
        }
        lw.rms_attn_ = std::move(*rms_attn);

        auto rms_ffn = load(p + ".post_attention_layernorm.weight", {d});
        if (!rms_ffn) {
            return std::unexpected(rms_ffn.error());
        }
        lw.rms_ffn_ = std::move(*rms_ffn);

        auto q = load(p + ".self_attn.q_proj.weight", {nq * hd, d});
        if (!q) {
            return std::unexpected(q.error());
        }

        auto k = load(p + ".self_attn.k_proj.weight", {nkv * hd, d});
        if (!k) {
            return std::unexpected(k.error());
        }

        auto v = load(p + ".self_attn.v_proj.weight", {nkv * hd, d});
        if (!v) {
            return std::unexpected(v.error());
        }

        const size_t q_elems =
            static_cast<size_t>(nq) * static_cast<size_t>(hd) * static_cast<size_t>(d);
        const size_t k_elems =
            static_cast<size_t>(nkv) * static_cast<size_t>(hd) * static_cast<size_t>(d);
        const size_t v_elems =
            static_cast<size_t>(nkv) * static_cast<size_t>(hd) * static_cast<size_t>(d);

        const size_t qkv_elems = q_elems + k_elems + v_elems;

        lw.qkv_ = DeviceBuffer<__nv_bfloat16>(qkv_elems);

        auto copy_q = cuda_check(cudaMemcpy(
            lw.qkv_.get(), q->get(), q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        if (!copy_q) {
            return std::unexpected(copy_q.error());
        }

        auto copy_k =
            cuda_check(cudaMemcpy(lw.qkv_.get() + q_elems, k->get(),
                                  k_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        if (!copy_k) {
            return std::unexpected(copy_k.error());
        }

        auto copy_v =
            cuda_check(cudaMemcpy(lw.qkv_.get() + q_elems + k_elems, v->get(),
                                  v_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        if (!copy_v) {
            return std::unexpected(copy_v.error());
        }

        w.layers_.push_back(std::move(lw));
    }

    return w;
}

}  // namespace engine
}  // namespace ccinfer
