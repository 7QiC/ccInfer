#pragma once

#include <memory>

#include <boost/asio/experimental/channel.hpp>
#include <boost/system/error_code.hpp>

#include "base/result.h"
#include "base/types.h"

namespace ccinfer {

namespace asio = boost::asio;

using BatchChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<WorkerBatchResult>)>;
using TokenChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<GeneratedToken>)>;
using AdmitSequenceChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<AdmitSequenceResult>)>;
using SuspendSequenceChannel =
    asio::experimental::channel<void(boost::system::error_code, Result<SuspendSequenceResult>)>;
using VoidChannel = asio::experimental::channel<void(boost::system::error_code, Result<void>)>;

using BatchFuture = std::shared_ptr<BatchChannel>;

}  // namespace ccinfer
