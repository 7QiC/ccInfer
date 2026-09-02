#include "tokenizer/gguf_tokenizer.h"

#include <string>
#include <vector>

#include "tokenizer/byte_level_bpe_tokenizer.h"

namespace ccinfer {

Result<std::unique_ptr<Tokenizer>> create_tokenizer_from_gguf(const GGUFReader& reader) {
    auto tokens_meta = reader.metadata("tokenizer.ggml.tokens");
    if (!tokens_meta.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
    const auto* tokens_arr = tokens_meta->as_array();
    if (tokens_arr == nullptr) return std::unexpected(ErrorCode::ModelLoadFailed);

    std::vector<std::string> tokens;
    tokens.reserve(tokens_arr->size());
    for (const auto& v : *tokens_arr) {
        tokens.push_back(v.as_string());
    }

    std::vector<std::string> merges;
    if (auto merges_meta = reader.metadata("tokenizer.ggml.merges"); merges_meta.has_value()) {
        if (const auto* merges_arr = merges_meta->as_array(); merges_arr != nullptr) {
            merges.reserve(merges_arr->size());
            for (const auto& v : *merges_arr) {
                merges.push_back(v.as_string());
            }
        }
    }

    std::vector<int32_t> token_types;
    if (auto types_meta = reader.metadata("tokenizer.ggml.token_type"); types_meta.has_value()) {
        if (const auto* types_arr = types_meta->as_array(); types_arr != nullptr) {
            token_types.reserve(types_arr->size());
            for (const auto& v : *types_arr) {
                const auto id = v.as_i64();
                if (!id.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
                token_types.push_back(static_cast<int32_t>(*id));
            }
        }
    }

    int32_t eos = -1;
    if (auto v = reader.metadata("tokenizer.ggml.eos_token_id"); v.has_value()) {
        auto id = v->as_u64();
        if (!id.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
        eos = static_cast<int32_t>(*id);
    }

    int32_t pad = -1;
    if (auto v = reader.metadata("tokenizer.ggml.padding_token_id"); v.has_value()) {
        auto id = v->as_u64();
        if (!id.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
        pad = static_cast<int32_t>(*id);
    } else if (auto v = reader.metadata("tokenizer.ggml.pad_token_id"); v.has_value()) {
        auto id = v->as_u64();
        if (!id.has_value()) return std::unexpected(ErrorCode::ModelLoadFailed);
        pad = static_cast<int32_t>(*id);
    }

    auto tokenizer = std::make_unique<ByteLevelBpeTokenizer>();
    auto r = tokenizer->load_from_vocab_and_merges(tokens, merges, token_types, eos, pad);
    if (!r) return std::unexpected(r.error());
    return std::unique_ptr<Tokenizer>(std::move(tokenizer));
}

}  // namespace ccinfer
