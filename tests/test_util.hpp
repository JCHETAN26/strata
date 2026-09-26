#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include "strata/matrix.hpp"

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

}  // namespace strata::test
