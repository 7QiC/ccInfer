# Layered static libraries. The public `engine` target keeps existing consumers
# (server/tests) working while allowing finer-grained rebuilds and dependencies.

add_library(ccinfer_quant STATIC
    quant/q8_0/q8_0_reference.cpp
)
target_include_directories(ccinfer_quant
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(ccinfer_quant
    PUBLIC
        ccop::ccop
)

add_library(ccinfer_config STATIC
    config/model_config.cpp
    config/engine_config.cpp
)
target_include_directories(ccinfer_config
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

add_library(ccinfer_checkpoint STATIC
    checkpoint/checkpoint.cpp
    checkpoint/gguf/reader.cpp
    checkpoint/gguf/weight_source.cpp
    checkpoint/gguf/config_loader.cpp
    checkpoint/gguf/checkpoint.cpp
    checkpoint/huggingface/reader.cpp
    checkpoint/huggingface/weight_source.cpp
    checkpoint/huggingface/config_loader.cpp
    checkpoint/huggingface/checkpoint.cpp
)
target_include_directories(ccinfer_checkpoint
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(ccinfer_checkpoint
    PUBLIC
        ccinfer_config
        ccinfer_quant
        ccop::ccop
        nlohmann_json::nlohmann_json
)

add_library(ccinfer_core STATIC
    backend/cuda/backend_cuda.cpp
    runtime/tensor.cpp
)
target_include_directories(ccinfer_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(ccinfer_core
    PUBLIC
        ccop::ccop
    PRIVATE
        spdlog::spdlog
        CUDA::cublas
        CUDA::cudart
)

add_library(ccinfer_cache STATIC
    cache/block_pool.cpp
    cache/block_storage.cpp
    cache/prefix_cache.cpp
)
target_include_directories(ccinfer_cache
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(ccinfer_cache
    PUBLIC
        ccinfer_core
        ccop::ccop
        spdlog::spdlog
    PRIVATE
        CUDA::cudart
)

add_library(ccinfer_model STATIC
    model/registry.cpp
    model/rope/rope_cache.cpp
    model/qwen3/qwen3_model.cpp
    model/qwen3/qwen3_weights.cpp
    model/weight.cpp
)
target_include_directories(ccinfer_model
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(ccinfer_model
    PUBLIC
        ccinfer_config
        ccinfer_cache
        ccinfer_core
        ccinfer_quant
        nlohmann_json::nlohmann_json
        spdlog::spdlog
    PRIVATE
        CUDA::cublas
        CUDA::cudart
)

add_library(ccinfer_engine STATIC
    engine/engine_core.cpp
    executor/single_device_executor.cpp
    worker/worker.cpp
    scheduler/scheduler.cpp
)
target_include_directories(ccinfer_engine
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(ccinfer_engine
    PUBLIC
        ccinfer_model
        ccinfer_checkpoint
        ccinfer_cache
        ccinfer_core
        fmt::fmt
        nlohmann_json::nlohmann_json
        spdlog::spdlog
    PRIVATE
        CUDA::cudart
)

add_library(tokenizer STATIC
    tokenizer/byte_level_bpe_tokenizer.cpp
    tokenizer/gguf_tokenizer.cpp
)

target_include_directories(tokenizer
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(tokenizer
    PUBLIC
        ccinfer_checkpoint
        nlohmann_json::nlohmann_json
)

if(BUILD_SERVER)
    add_library(server STATIC
        http/http_server.cpp
    )

    target_include_directories(server
        PUBLIC
            ${CMAKE_SOURCE_DIR}/src
            ${Boost_INCLUDE_DIRS}
    )

    target_link_libraries(server
        PUBLIC
            ccinfer_engine
            tokenizer
            nlohmann_json::nlohmann_json
    )

    add_executable(ccinfer-server
        main.cpp
    )

    target_link_libraries(ccinfer-server
        PRIVATE
            server
            ccinfer_engine
    )
endif()
