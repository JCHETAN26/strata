cmake_minimum_required(VERSION 3.25)

project(strata VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(STRATA_BUILD_TESTS "Build unit tests" ON)
option(STRATA_BUILD_BENCHMARKS "Build benchmarks" ON)
option(STRATA_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(STRATA_TSAN "Build with ThreadSanitizer (exclusive with STRATA_SANITIZE)" OFF)

# SIMD kernels are selected at compile time. NEON is baseline on arm64; AVX2 + FMA must be enabled
# explicitly on x86_64 (on by default there; turn off for machines without AVX2).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
  set(STRATA_IS_X86_64 ON)
endif()
include(CMakeDependentOption)
cmake_dependent_option(STRATA_AVX2 "Compile AVX2 + FMA kernels (x86_64)" ON "STRATA_IS_X86_64" OFF)

# Warnings and sanitizers apply to every Strata target through this interface library.
add_library(strata_options INTERFACE)
target_compile_options(strata_options INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
if(STRATA_SANITIZE)
  target_compile_options(strata_options INTERFACE
    -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  target_link_options(strata_options INTERFACE -fsanitize=address,undefined)
endif()
if(STRATA_TSAN)
  target_compile_options(strata_options INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(strata_options INTERFACE -fsanitize=thread)
endif()

find_package(tl-expected CONFIG REQUIRED)

add_library(strata STATIC
  src/distance/avx2.cpp
  src/distance/dispatch.cpp
  src/distance/neon.cpp
  src/distance/scalar.cpp
  src/eval/recall.cpp
  src/index/brute_force.cpp
  src/storage/collection.cpp
  src/storage/crc32c.cpp
  src/storage/dataset.cpp
  src/storage/file.cpp
  src/storage/snapshot.cpp
  src/storage/wal.cpp
  src/util/thread_pool.cpp)
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
find_package(Threads REQUIRED)
target_link_libraries(strata
  PUBLIC tl::expected Threads::Threads
  PRIVATE strata_options)
if(STRATA_AVX2)
  # PUBLIC: headers test __AVX2__ to declare the avx2:: kernels, so every consumer needs the flags.
  target_compile_options(strata PUBLIC -mavx2 -mfma)
endif()

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(STRATA_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
