#include "strata/kmeans.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <string>

#include "strata/distance.hpp"

namespace strata {

namespace {

// Index and squared distance of the nearest centroid.
std::pair<std::uint32_t, float> nearest(std::span<const float> x, const Matrix<float>& centroids,
                                        DistanceFn l2) {
  std::uint32_t best = 0;
  float best_d = std::numeric_limits<float>::max();
  for (std::size_t c = 0; c < centroids.rows(); ++c) {
    const float d = l2(x, centroids.row(c));
    if (d < best_d) {
      best_d = d;
      best = static_cast<std::uint32_t>(c);
    }
  }
  return {best, best_d};
}

// k-means++: first centroid uniform, each next one sampled with probability proportional to
// squared distance from the nearest centroid chosen so far.
Matrix<float> init_plus_plus(const Matrix<float>& data, std::size_t k, std::mt19937_64& rng,
                             DistanceFn l2) {
  const std::size_t n = data.rows();
  Matrix<float> centroids(k, data.cols());
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  std::ranges::copy(data.row(pick(rng)), centroids.row(0).begin());
  std::vector<double> d2(n);
  for (std::size_t i = 0; i < n; ++i) {
    d2[i] = l2(data.row(i), centroids.row(0));
  }
  for (std::size_t c = 1; c < k; ++c) {
    double total = 0;
    for (double d : d2) {
      total += d;
    }
    std::size_t chosen = pick(rng);  // fallback when every point coincides with a centroid
    if (total > 0) {
      std::uniform_real_distribution<double> u(0.0, total);
      double target = u(rng);
      for (std::size_t i = 0; i < n; ++i) {
        target -= d2[i];
        if (target <= 0) {
          chosen = i;
          break;
        }
      }
    }
    std::ranges::copy(data.row(chosen), centroids.row(c).begin());
    for (std::size_t i = 0; i < n; ++i) {
      d2[i] = std::min(d2[i], static_cast<double>(l2(data.row(i), centroids.row(c))));
    }
  }
  return centroids;
}

}  // namespace

Expected<KMeansResult> kmeans(const Matrix<float>& data, const KMeansParams& params,
                              ThreadPool* pool) {
  const std::size_t n = data.rows();
  const std::size_t dim = data.cols();
  const std::size_t k = params.k;
  if (k == 0 || n == 0 || dim == 0) {
    return make_error(ErrorCode::kInvalidArgument, "k-means needs k > 0 and non-empty data");
  }
  if (n < k) {
    return make_error(
        ErrorCode::kInvalidArgument,
        "k-means needs at least k = " + std::to_string(k) + " points, got " + std::to_string(n));
  }
  const DistanceFn l2 = distance_function(Metric::kL2);
  std::mt19937_64 rng(params.seed);

  KMeansResult result;
  result.centroids = init_plus_plus(data, k, rng, l2);
  result.assignment.assign(n, 0);
  std::vector<float> dist(n);
  double previous = std::numeric_limits<double>::max();

  auto for_each_point = [&](const std::function<void(std::size_t)>& fn) {
    if (pool != nullptr) {
      pool->parallel_for(n, fn);
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        fn(i);
      }
    }
  };

  for (std::size_t iter = 0; iter < params.max_iterations; ++iter) {
    // Assignment step.
    for_each_point([&](std::size_t i) {
      const auto [c, d] = nearest(data.row(i), result.centroids, l2);
      result.assignment[i] = c;
      dist[i] = d;
    });
    double inertia = 0;
    for (float d : dist) {
      inertia += d;
    }
    result.inertia = inertia;
    result.iterations = iter + 1;

    // Update step: centroid = mean of its points (accumulated in double).
    std::vector<double> sums(k * dim, 0.0);
    std::vector<std::size_t> counts(k, 0);
    for (std::size_t i = 0; i < n; ++i) {
      const auto c = result.assignment[i];
      ++counts[c];
      const auto row = data.row(i);
      for (std::size_t j = 0; j < dim; ++j) {
        sums[c * dim + j] += row[j];
      }
    }
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) {
        continue;
      }
      auto centroid = result.centroids.row(c);
      for (std::size_t j = 0; j < dim; ++j) {
        centroid[j] = static_cast<float>(sums[c * dim + j] / static_cast<double>(counts[c]));
      }
    }
    // Empty clusters: split the largest cluster by moving the empty centroid to one of its
    // points, so the next assignment step divides that cluster.
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] != 0) {
        continue;
      }
      const auto largest = static_cast<std::uint32_t>(
          std::distance(counts.begin(), std::ranges::max_element(counts)));
      if (counts[largest] < 2) {
        break;  // fewer distinct points than clusters; nothing left to split
      }
      // Farthest point of the largest cluster: splits it along its widest direction.
      std::size_t far = 0;
      float far_d = -1;
      for (std::size_t i = 0; i < n; ++i) {
        if (result.assignment[i] == largest && dist[i] > far_d) {
          far_d = dist[i];
          far = i;
        }
      }
      std::ranges::copy(data.row(far), result.centroids.row(c).begin());
      result.assignment[far] = static_cast<std::uint32_t>(c);
      --counts[largest];
      counts[c] = 1;
    }

    if (previous - inertia <= params.tolerance * previous) {
      break;
    }
    previous = inertia;
  }
  return result;
}

}  // namespace strata
