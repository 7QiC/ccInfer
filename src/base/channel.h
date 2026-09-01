#pragma once

#include <memory>

#include <boost/asio/experimental/channel.hpp>
#include <boost/system/error_code.hpp>

#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

using BatchChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<BatchResult>)>;

using BatchFuture = std::shared_ptr<BatchChannel>;

}  // namespace ccinfer
