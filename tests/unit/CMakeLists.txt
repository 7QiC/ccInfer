function(ccinfer_add_unit_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ccinfer_engine gtest_main ${ARGN})
    gtest_discover_tests(${name})
endfunction()

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

ccinfer_add_unit_test(test_batch_translator CUDA::cudart)
ccinfer_add_unit_test(test_scheduler)
ccinfer_add_unit_test(test_prefix_cache)
ccinfer_add_unit_test(test_rope_cache CUDA::cudart)
