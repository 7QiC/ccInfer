function(ccinfer_add_unit_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ccinfer_engine gtest_main ${ARGN})
    gtest_discover_tests(${name})
endfunction()

add_executable(test_q8_0 test_q8_0.cpp)
target_link_libraries(test_q8_0 PRIVATE ccinfer_quant gtest_main)
gtest_discover_tests(test_q8_0)

ccinfer_add_unit_test(test_precision)

add_executable(test_gguf_reader test_gguf_reader.cpp)
target_link_libraries(test_gguf_reader PRIVATE ccinfer_checkpoint gtest_main)
target_compile_definitions(test_gguf_reader PRIVATE
    GGUF_ARTIFACT_PATH="${CMAKE_SOURCE_DIR}/models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf")
gtest_discover_tests(test_gguf_reader)

add_executable(test_reader test_reader.cpp)
target_link_libraries(test_reader PRIVATE ccinfer_checkpoint gtest_main)
gtest_discover_tests(test_reader)

add_executable(test_weight_source test_weight_source.cpp)
target_link_libraries(test_weight_source PRIVATE ccinfer_checkpoint gtest_main)
gtest_discover_tests(test_weight_source)

add_executable(test_gguf_config_loader test_gguf_config_loader.cpp)
target_link_libraries(test_gguf_config_loader PRIVATE ccinfer_checkpoint gtest_main)
gtest_discover_tests(test_gguf_config_loader)

add_executable(test_checkpoint test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE ccinfer_checkpoint gtest_main)
gtest_discover_tests(test_checkpoint)

ccinfer_add_unit_test(test_result)
ccinfer_add_unit_test(test_block)
ccinfer_add_unit_test(test_block_pool)
ccinfer_add_unit_test(test_block_storage CUDA::cudart)
ccinfer_add_unit_test(test_device_buffer CUDA::cudart)
ccinfer_add_unit_test(test_tensor CUDA::cudart)
ccinfer_add_unit_test(test_config)
ccinfer_add_unit_test(test_smoke CUDA::cudart)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE tokenizer gtest_main)
gtest_discover_tests(test_tokenizer)

add_executable(test_tokenizer_alignment test_tokenizer_alignment.cpp)
target_link_libraries(test_tokenizer_alignment PRIVATE tokenizer gtest_main)
target_compile_definitions(test_tokenizer_alignment PRIVATE
    REFERENCE_PATH="${CMAKE_SOURCE_DIR}/tests/unit/tokenizer_alignment_reference.json")
gtest_discover_tests(test_tokenizer_alignment)

add_executable(test_tokenizer_qwen35 test_tokenizer_qwen35.cpp)
target_link_libraries(test_tokenizer_qwen35 PRIVATE tokenizer gtest_main)
target_compile_definitions(test_tokenizer_qwen35 PRIVATE
    GGUF_ARTIFACT_PATH="${CMAKE_SOURCE_DIR}/models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf"
    TOKENIZER_JSON_PATH="${CMAKE_SOURCE_DIR}/models/qwen3_5-2B/tokenizer.json")
gtest_discover_tests(test_tokenizer_qwen35)

ccinfer_add_unit_test(test_worker CUDA::cudart)
ccinfer_add_unit_test(test_scheduler)
ccinfer_add_unit_test(test_prefix_cache)
ccinfer_add_unit_test(test_rope_cache CUDA::cudart)
