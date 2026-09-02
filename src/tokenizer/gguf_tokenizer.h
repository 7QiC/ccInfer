#pragma once

#include <memory>

#include "base/error.h"
#include "checkpoint/gguf/reader.h"
#include "tokenizer/tokenizer.h"

namespace ccinfer {

Result<std::unique_ptr<Tokenizer>> create_tokenizer_from_gguf(const GGUFReader& reader);

}  // namespace ccinfer
