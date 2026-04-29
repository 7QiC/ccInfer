add_executable(test_result unit/test_result.cpp)
target_link_libraries(test_result PRIVATE engine gtest_main)
include(GoogleTest)
gtest_discover_tests(test_result)

add_executable(test_device_buffer unit/test_device_buffer.cpp)
target_link_libraries(test_device_buffer PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_device_buffer)

add_executable(test_tensor unit/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine gtest_main)
gtest_discover_tests(test_tensor)
