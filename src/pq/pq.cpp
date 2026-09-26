#include "strata/pq.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <string>

#include "strata/kmeans.hpp"

namespace strata {

namespace {

// Copy of `v`, L2-normalized (zero vectors stay zero).
void normalize_into(std::span<const float> v, std::span<float> out) {
  double norm = 0;
  for (float x : v) {
    norm += static_cast<double>(x) * x;
  }
  const auto scale = norm > 0 ? static_cast<float>(1.0 / std::sqrt(norm)) : 0.0F;
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = v[i] * scale;
  }
}

// Bounded max-heap top-k over (distance, id).
void push_top(std::vector<Neighbor>& heap, std::size_t k, Neighbor candidate) {
  if (heap.size() < k) {
    heap.push_back(candidate);
    std::push_heap(heap.begin(), heap.end());
  } else if (candidate < heap.front()) {
    std::pop_heap(heap.begin(), heap.end());
    heap.back() = candidate;
    std::push_heap(heap.begin(), heap.end());
  }
}

}  // namespace

Expected<ProductQuantizer> ProductQuantizer::train(const Matrix<float>& data, Metric metric,
                                                   const PqParams& params, ThreadPool* pool) {
  const std::size_t dim = data.cols();
  const std::size_t m = params.m;
  if (m == 0 || dim == 0 || dim % m != 0) {
    return make_error(
        ErrorCode::kInvalidArgument,
        "dimension " + std::to_string(dim) + " is not divisible by m = " + std::to_string(m));
  }
  if (data.rows() < kCentroids) {
    return make_error(ErrorCode::kInvalidArgument,
                      "PQ training needs at least 256 rows, got " + std::to_string(data.rows()));
  }

  // Sample training rows (without replacement) if there are more than requested.
  std::vector<std::size_t> rows(data.rows());
  std::iota(rows.begin(), rows.end(), 0);
  if (rows.size() > params.max_training_rows) {
    std::mt19937_64 rng(params.seed);
    std::shuffle(rows.begin(), rows.end(), rng);
    rows.resize(std::max(params.max_training_rows, kCentroids));
  }
  Matrix<float> training(rows.size(), dim);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (metric == Metric::kCosine) {
      normalize_into(data.row(rows[i]), training.row(i));
    } else {
      std::ranges::copy(data.row(rows[i]), training.row(i).begin());
    }
  }

  const std::size_t sub_dim = dim / m;
  Matrix<float> codebooks(m * kCentroids, sub_dim);
  Matrix<float> sub(training.rows(), sub_dim);
  for (std::size_t s = 0; s < m; ++s) {
    for (std::size_t i = 0; i < training.rows(); ++i) {
      const auto row = training.row(i).subspan(s * sub_dim, sub_dim);
      std::ranges::copy(row, sub.row(i).begin());
    }
    auto result = kmeans(
        sub, {.k = kCentroids, .max_iterations = params.kmeans_iterations, .seed = params.seed + s},
        pool);
    if (!result) {
      return tl::unexpected(result.error());
    }
    for (std::size_t c = 0; c < kCentroids; ++c) {
      std::ranges::copy(result->centroids.row(c), codebooks.row(s * kCentroids + c).begin());
    }
  }
  return ProductQuantizer(dim, m, metric, std::move(codebooks));
}

std::span<const float> ProductQuantizer::centroid(std::size_t sub, std::size_t c) const noexcept {
  return codebooks_.row(sub * kCentroids + c);
}

void ProductQuantizer::encode(std::span<const float> vector, std::span<std::uint8_t> code) const {
  assert(vector.size() == dim_ && code.size() == m_);
  std::vector<float> normalized;
  if (metric_ == Metric::kCosine) {
    normalized.resize(dim_);
    normalize_into(vector, normalized);
    vector = normalized;
  }
  const DistanceFn l2 = distance_function(Metric::kL2);
  const std::size_t sub_dim = dim_ / m_;
  for (std::size_t s = 0; s < m_; ++s) {
    const auto x = vector.subspan(s * sub_dim, sub_dim);
    std::size_t best = 0;
    float best_d = l2(x, centroid(s, 0));
    for (std::size_t c = 1; c < kCentroids; ++c) {
      const float d = l2(x, centroid(s, c));
      if (d < best_d) {
        best_d = d;
        best = c;
      }
    }
    code[s] = static_cast<std::uint8_t>(best);
  }
}

void ProductQuantizer::decode(std::span<const std::uint8_t> code, std::span<float> out) const {
  assert(code.size() == m_ && out.size() == dim_);
  const std::size_t sub_dim = dim_ / m_;
  for (std::size_t s = 0; s < m_; ++s) {
    std::ranges::copy(centroid(s, code[s]), out.begin() + static_cast<std::ptrdiff_t>(s * sub_dim));
  }
}

void ProductQuantizer::compute_table(std::span<const float> query, std::span<float> table) const {
  assert(query.size() == dim_ && table.size() == m_ * kCentroids);
  std::vector<float> normalized;
  if (metric_ == Metric::kCosine) {
    normalized.resize(dim_);
    normalize_into(query, normalized);
    query = normalized;
  }
  // L2 decomposes into per-subspace squared distances; IP and cosine into per-subspace negated
  // dot products (cosine adds the constant 1 in table_distance).
  const DistanceFn partial =
      distance_function(metric_ == Metric::kL2 ? Metric::kL2 : Metric::kInnerProduct);
  const std::size_t sub_dim = dim_ / m_;
  for (std::size_t s = 0; s < m_; ++s) {
    const auto q = query.subspan(s * sub_dim, sub_dim);
    for (std::size_t c = 0; c < kCentroids; ++c) {
      table[s * kCentroids + c] = partial(q, centroid(s, c));
    }
  }
}

float ProductQuantizer::table_distance(std::span<const float> table,
                                       std::span<const std::uint8_t> code) const noexcept {
  // Four independent sums: with one, every lookup waits for the previous add (a latency chain of
  // m float adds). Measured on SIFT10K ADC scans: see docs/devlog.md, Phase 5.
  const float* t = table.data();
  const std::uint8_t* c = code.data();
  float s0 = 0.0F;
  float s1 = 0.0F;
  float s2 = 0.0F;
  float s3 = 0.0F;
  std::size_t s = 0;
  for (; s + 4 <= m_; s += 4, t += 4 * kCentroids) {
    s0 += t[c[s]];
    s1 += t[kCentroids + c[s + 1]];
    s2 += t[(2 * kCentroids) + c[s + 2]];
    s3 += t[(3 * kCentroids) + c[s + 3]];
  }
  for (; s < m_; ++s, t += kCentroids) {
    s0 += t[c[s]];
  }
  const float sum = (s0 + s1) + (s2 + s3);
  return metric_ == Metric::kCosine ? 1.0F + sum : sum;
}

Expected<PqIndex> PqIndex::create(ProductQuantizer quantizer, bool keep_originals) {
  return PqIndex(std::move(quantizer), keep_originals);
}

Expected<void> PqIndex::add_batch(const Matrix<float>& vectors, ThreadPool* pool) {
  if (vectors.empty()) {
    return {};
  }
  if (vectors.cols() != pq_.dim()) {
    return make_error(ErrorCode::kDimensionMismatch, "expected dimension " +
                                                         std::to_string(pq_.dim()) + ", got " +
                                                         std::to_string(vectors.cols()));
  }
  const std::size_t m = pq_.code_size();
  const std::size_t first = size_;
  codes_.resize((size_ + vectors.rows()) * m);
  auto encode_row = [&](std::size_t i) {
    pq_.encode(vectors.row(i), std::span(codes_).subspan((first + i) * m, m));
  };
  if (pool != nullptr) {
    pool->parallel_for(vectors.rows(), encode_row);
  } else {
    for (std::size_t i = 0; i < vectors.rows(); ++i) {
      encode_row(i);
    }
  }
  if (keep_originals_) {
    const auto values = vectors.data();
    originals_.insert(originals_.end(), values.begin(), values.end());
  }
  size_ += vectors.rows();
  return {};
}

Expected<std::vector<Neighbor>> PqIndex::search(std::span<const float> query, std::size_t k,
                                                const PqSearchParams& params) const {
  if (query.size() != pq_.dim()) {
    return make_error(ErrorCode::kDimensionMismatch, "expected dimension " +
                                                         std::to_string(pq_.dim()) + ", got " +
                                                         std::to_string(query.size()));
  }
  if (params.rerank > 0 && !keep_originals_) {
    return make_error(ErrorCode::kInvalidArgument, "reranking needs keep_originals");
  }
  k = std::min(k, size_);
  if (k == 0) {
    return std::vector<Neighbor>{};
  }
  const std::size_t candidates = std::min(std::max(params.rerank, k), size_);

  std::vector<float> table(pq_.m() * ProductQuantizer::kCentroids);
  pq_.compute_table(query, table);
  std::vector<Neighbor> heap;
  heap.reserve(candidates + 1);
  const std::size_t m = pq_.code_size();
  const std::uint8_t* code = codes_.data();
  for (std::size_t i = 0; i < size_; ++i, code += m) {
    push_top(heap, candidates, {static_cast<VectorId>(i), pq_.table_distance(table, {code, m})});
  }

  if (params.rerank > 0) {
    const DistanceFn exact = distance_function(pq_.metric());
    const std::size_t dim = pq_.dim();
    for (auto& n : heap) {
      n.distance = exact(query, {originals_.data() + static_cast<std::size_t>(n.id) * dim, dim});
    }
  }
  std::ranges::sort(heap);
  heap.resize(k);
  return heap;
}

}  // namespace strata
