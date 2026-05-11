#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/result.h"

namespace ccinfer {
namespace server {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual Result<void> load(const std::string& path) = 0;

    virtual Result<std::vector<int32_t>> encode(std::string_view text,
                                                bool add_special_tokens = false) const = 0;

    virtual Result<std::string> decode(const std::vector<int32_t>& token_ids,
                                       bool skip_special_tokens = true) const = 0;

    virtual int32_t bos_token_id() const noexcept = 0;
    virtual int32_t eos_token_id() const noexcept = 0;
    virtual int32_t pad_token_id() const noexcept = 0;
    virtual int32_t unk_token_id() const noexcept = 0;

    virtual int32_t vocab_size() const noexcept = 0;
};

Result<std::unique_ptr<Tokenizer>> create_tokenizer(const std::string& model_dir);

}  // namespace server
}  // namespace ccinfer