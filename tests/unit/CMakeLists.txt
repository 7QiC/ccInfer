add_executable(test_result test_result.cpp)
target_link_libraries(test_result PRIVATE engine gtest_main)
gtest_discover_tests(test_result)

add_executable(test_block test_block.cpp)
target_link_libraries(test_block PRIVATE engine gtest_main)
gtest_discover_tests(test_block)

add_executable(test_kv_cache_storage test_kv_cache_storage.cpp)
target_link_libraries(test_kv_cache_storage PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_kv_cache_storage)

add_executable(test_kv_cache_manager test_kv_cache_manager.cpp)
target_link_libraries(test_kv_cache_manager PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_kv_cache_manager)

add_executable(test_device_buffer test_device_buffer.cpp)
target_link_libraries(test_device_buffer PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_device_buffer)

add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine gtest_main)
gtest_discover_tests(test_tensor)

add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE engine gtest_main)
gtest_discover_tests(test_config)

add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_smoke)

add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE engine server gtest_main)
gtest_discover_tests(test_tokenizer)

add_executable(test_tokenizer_alignment test_tokenizer_alignment.cpp)
target_link_libraries(test_tokenizer_alignment PRIVATE engine server gtest_main)
target_compile_definitions(test_tokenizer_alignment PRIVATE
    REFERENCE_PATH="${CMAKE_SOURCE_DIR}/tests/unit/tokenizer_alignment_reference.json")
gtest_discover_tests(test_tokenizer_alignment)

add_executable(test_batch_translator test_batch_translator.cpp)
target_link_libraries(test_batch_translator PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_batch_translator)
