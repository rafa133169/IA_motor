cmake_minimum_required(VERSION 3.16)
project(NevronCpp VERSION 0.1 LANGUAGES CXX)

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------
# LIBRERÍA PRINCIPAL
# -------------------------
add_library(nevrocpp
    src/core/Tensor.cpp
    src/layers/ReLU.cpp
    src/losses/MSE.cpp
)

target_include_directories(nevrocpp
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# -------------------------
# EJEMPLOS
# -------------------------
add_executable(basic_usage examples/basic_usage.cpp)
target_link_libraries(basic_usage nevrocpp)

add_executable(basic_training examples/basic_training.cpp)
target_link_libraries(basic_training nevrocpp)

add_executable(polymorphic_usage examples/polymorphic_usage.cpp)
target_link_libraries(polymorphic_usage nevrocpp)

# -------------------------
# TESTING
# -------------------------
enable_testing()
add_subdirectory(tests)
