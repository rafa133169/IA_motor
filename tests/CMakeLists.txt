include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(test_relu test_relu.cpp)
target_link_libraries(test_relu gtest_main nevrocpp)

include(GoogleTest)
gtest_discover_tests(test_relu)
