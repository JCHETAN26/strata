#include "strata/distance.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace strata {
namespace {

TEST(Distance, L2SquaredKnownValue) {
  const std::vector<float> a{1, 2, 3};
  const std::vector<float> b{4, 6, 3};
  EXPECT_FLOAT_EQ(scalar::l2_squared(a, b), 9.0F + 16.0F);
}

TEST(Distance, L2SquaredOfIdenticalVectorsIsZero) {
  const std::vector<float> a{0.5F, -1.25F, 3.0F};
  EXPECT_EQ(scalar::l2_squared(a, a), 0.0F);
}

TEST(Distance, L2IsSymmetric) {
  const std::vector<float> a{1, -2, 3, 0.5F};
  const std::vector<float> b{-4, 6, 1, 2};
  EXPECT_EQ(scalar::l2_squared(a, b), scalar::l2_squared(b, a));
}

TEST(Distance, InnerProductIsNegatedDot) {
  const std::vector<float> a{1, 2, 3};
  const std::vector<float> b{4, -5, 6};
  EXPECT_FLOAT_EQ(scalar::inner_product(a, b), -(4.0F - 10.0F + 18.0F));
}

TEST(Distance, CosineOfParallelVectorsIsZero) {
  const std::vector<float> a{1, 2, 3};
  const std::vector<float> b{2, 4, 6};
  EXPECT_NEAR(scalar::cosine_distance(a, b), 0.0F, 1e-6F);
}

TEST(Distance, CosineOfOrthogonalVectorsIsOne) {
  const std::vector<float> a{1, 0};
  const std::vector<float> b{0, 5};
  EXPECT_FLOAT_EQ(scalar::cosine_distance(a, b), 1.0F);
}

TEST(Distance, CosineOfOppositeVectorsIsTwo) {
  const std::vector<float> a{1, 1};
  const std::vector<float> b{-3, -3};
  EXPECT_NEAR(scalar::cosine_distance(a, b), 2.0F, 1e-6F);
}

TEST(Distance, CosineWithZeroVectorIsOne) {
  const std::vector<float> zero{0, 0, 0};
  const std::vector<float> b{1, 2, 3};
  EXPECT_EQ(scalar::cosine_distance(zero, b), 1.0F);
  EXPECT_EQ(scalar::cosine_distance(zero, zero), 1.0F);
}

TEST(Distance, EmptyVectors) {
  const std::vector<float> empty;
  EXPECT_EQ(scalar::l2_squared(empty, empty), 0.0F);
  EXPECT_EQ(scalar::inner_product(empty, empty), 0.0F);
  EXPECT_EQ(scalar::cosine_distance(empty, empty), 1.0F);
}

TEST(Distance, DispatchMatchesScalar) {
  const std::vector<float> a{1, 2, 3};
  const std::vector<float> b{-1, 0.5F, 2};
  EXPECT_EQ(distance(Metric::kL2, a, b), scalar::l2_squared(a, b));
  EXPECT_EQ(distance(Metric::kInnerProduct, a, b), scalar::inner_product(a, b));
  EXPECT_EQ(distance(Metric::kCosine, a, b), scalar::cosine_distance(a, b));
}

TEST(Distance, MetricNamesRoundTrip) {
  for (Metric m : {Metric::kL2, Metric::kInnerProduct, Metric::kCosine}) {
    EXPECT_EQ(parse_metric(to_string(m)), m);
  }
  EXPECT_EQ(parse_metric("angular"), Metric::kCosine);
  EXPECT_EQ(parse_metric("hamming"), std::nullopt);
}

}  // namespace
}  // namespace strata
