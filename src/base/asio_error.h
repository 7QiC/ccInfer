#pragma once

#include <boost/asio/error.hpp>
#include <boost/system/error_code.hpp>

#include "base/error_code.h"

namespace ccinfer {

inline ErrorCode to_error_code(const boost::system::error_code& ec) {
    if (!ec) return ErrorCode::Ok;
    if (ec == boost::asio::error::operation_aborted) return ErrorCode::ChannelCancelled;
    if (ec == boost::asio::error::eof) return ErrorCode::ChannelClosed;
    return ErrorCode::ChannelError;
}

}  // namespace ccinfer
