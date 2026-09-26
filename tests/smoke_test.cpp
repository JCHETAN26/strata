#include <gtest/gtest.h>

#include "strata/version.hpp"

// Verifies the toolchain, GoogleTest, and include paths are wired up.
TEST(Smoke, VersionIsConsistent) {
  EXPECT_EQ(strata::kVersionString, "0.1.0");
  EXPECT_EQ(strata::kVersionMajor, 0);
  EXPECT_EQ(strata::kVersionMinor, 1);
}
