#pragma once

#include <boost/asio/experimental/channel.hpp>
#include <boost/system/error_code.hpp>

#include "common/error_code.h"
#include "common/result.h"
#include "common/types.h"

namespace ccinfer {

namespace asio = boost::asio;

using SeqIdChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<CreateSequenceResult>)>;
using VoidChannel = asio::experimental::channel<void(boost::system::error_code, Result<void>)>;
using BatchChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<BatchResult>)>;
using TokenChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<GeneratedToken>)>;

}  // namespace ccinfer
