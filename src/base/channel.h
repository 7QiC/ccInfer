#pragma once

#include <boost/asio/experimental/channel.hpp>
#include <boost/system/error_code.hpp>

#include "base/error_code.h"
#include "base/result.h"
#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

using BatchChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<WorkerBatchResult>)>;
using TokenChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<GeneratedToken>)>;

}  // namespace ccinfer
