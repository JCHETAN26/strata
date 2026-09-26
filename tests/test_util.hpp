#pragma once

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <random>
#include <string>

#include "strata/distance.hpp"
#include "strata/matrix.hpp"

namespace strata {

// gtest prints parameters it has no printer for as raw bytes ("4-byte object <00-00 00-00>"),
// and CMake's gtest_discover_tests builds ctest names from that printed value. Every
// parameter type used in a parameterized suite gets a PrintTo (found by ADL) instead.
inline void PrintTo(Metric metric, std::ostream* os) { *os << to_string(metric); }

}  // namespace strata

namespace strata::test {

// Uniform [-1, 1) vectors from a fixed seed, so failures reproduce.
inline Matrix<float> random_matrix(std::size_t rows, std::size_t cols, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  Matrix<float> m(rows, cols);
  for (float& x : m.data()) {
    x = dist(rng);
  }
  return m;
}

// Name generator for INSTANTIATE_TEST_SUITE_P: the parameter's PrintTo output. Using the same
// string for the gtest name and the printed value (which ctest names are built from) keeps both
// readable and identical on every platform. The printed value must be a valid test name
// (letters, digits, underscores).
struct PrintedName {
  template <typename T>
  std::string operator()(const ::testing::TestParamInfo<T>& info) const {
    return ::testing::PrintToString(info.param);
  }
};

}  // namespace strata::test
