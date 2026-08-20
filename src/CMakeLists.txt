add_library(engine STATIC
    backend/cuda/backend_cuda.cpp
    core/tensor.cpp
    executor/single_device_executor.cpp
    worker/worker.cpp
    worker/batch_translator.cpp
    model/rope/rope_cache.cpp
    cache/kv_cache_manager.cpp
    cache/kv_cache_storage.cpp
    cache/prefix_cache.cpp
    model/loader.cpp
    model/registry.cpp
    model/qwen3/qwen3_model.cpp
    model/qwen3/qwen3_weights.cpp
    scheduler/scheduler.cpp
)

target_include_directories(engine
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(engine
    PUBLIC
        ccop::ccop
        nlohmann_json::nlohmann_json
        fmt::fmt
        spdlog::spdlog
    PRIVATE
        CUDA::cublas
        CUDA::cudart
)

if(BUILD_SERVER)
    add_library(server STATIC
        http/http_server.cpp
        tokenizer/byte_level_bpe_tokenizer.cpp
    )

    target_include_directories(server
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
            ${Boost_INCLUDE_DIRS}
    )

    target_link_libraries(server
        PUBLIC
            engine
    )

    add_executable(ccinfer-server
        main.cpp
    )

    target_link_libraries(ccinfer-server
        PRIVATE
            server
            engine
    )
endif()
