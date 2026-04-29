add_executable(test_result unit/test_result.cpp)
target_link_libraries(test_result PRIVATE engine gtest_main)
include(GoogleTest)
gtest_discover_tests(test_result)
