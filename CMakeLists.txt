cmake_minimum_required(VERSION 3.25)

project(strata VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(STRATA_BUILD_TESTS "Build unit tests" ON)
option(STRATA_BUILD_BENCHMARKS "Build benchmarks" ON)
option(STRATA_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

# Warnings and sanitizers apply to every Strata target through this interface library.
add_library(strata_options INTERFACE)
target_compile_options(strata_options INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
if(STRATA_SANITIZE)
  target_compile_options(strata_options INTERFACE
    -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  target_link_options(strata_options INTERFACE -fsanitize=address,undefined)
endif()

# Core library. Header-only until the first source file lands in src/.
add_library(strata INTERFACE)
add_library(strata::strata ALIAS strata)
target_include_directories(strata INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(strata INTERFACE cxx_std_20)

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(STRATA_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
