#pragma once

#include <boost/asio/experimental/channel.hpp>

#include "common/error_code.h"
#include "common/types.h"

namespace ccinfer {

namespace asio = boost::asio;

using SeqIdChannel = asio::experimental::channel<void(ErrorCode, SequenceId)>;
using VoidChannel = asio::experimental::channel<void(ErrorCode)>;
using BatchChannel = asio::experimental::channel<void(ErrorCode, BatchResult)>;

}  // namespace ccinfer
