#pragma once

#include <boost/asio/error.hpp>
#include <boost/asio/experimental/channel_error.hpp>
#include <boost/system/error_code.hpp>

#include "common/error_code.h"

namespace ccinfer {

inline ErrorCode to_error_code(const boost::system::error_code& ec) {
    if (!ec) return ErrorCode::Ok;
    if (ec == boost::asio::error::operation_aborted ||
        ec == boost::asio::experimental::error::channel_cancelled)
        return ErrorCode::ChannelCancelled;
    if (ec == boost::asio::error::eof ||
        ec == boost::asio::experimental::error::channel_closed)
        return ErrorCode::ChannelClosed;
    return ErrorCode::ChannelError;
}

}  // namespace ccinfer
