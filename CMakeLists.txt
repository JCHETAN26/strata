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

find_package(tl-expected CONFIG REQUIRED)

add_library(strata STATIC
  src/distance/scalar.cpp
  src/eval/recall.cpp
  src/index/brute_force.cpp
  src/storage/dataset.cpp)
add_library(strata::strata ALIAS strata)

# The HNSW implementation is written by hand (see CLAUDE.md). Until src/index/hnsw.cpp exists,
# the library, tests, and harness build without it.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/index/hnsw.cpp")
  set(STRATA_HAS_HNSW ON)
  target_sources(strata PRIVATE src/index/hnsw.cpp)
  target_compile_definitions(strata PUBLIC STRATA_HAS_HNSW=1)
else()
  set(STRATA_HAS_HNSW OFF)
  message(STATUS "src/index/hnsw.cpp not found: building without HNSW")
endif()
target_include_directories(strata PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(strata PUBLIC cxx_std_20)
target_link_libraries(strata
  PUBLIC tl::expected
  PRIVATE strata_options)

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(STRATA_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
