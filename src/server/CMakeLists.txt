add_library(server STATIC
    http/http_server.cpp
    scheduler/scheduler.cpp
    server.cpp
)

target_include_directories(server
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(server
    PUBLIC
        engine
)
