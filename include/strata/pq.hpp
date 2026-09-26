#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "strata/distance.hpp"
#include "strata/error.hpp"
#include "strata/matrix.hpp"
#include "strata/thread_pool.hpp"
#include "strata/types.hpp"

namespace strata {

struct PqParams {
  // Number of sub-quantizers. Each vector is split into m sub-vectors of dim / m values, and each
  // sub-vector is stored as one byte (the index of its nearest of 256 centroids).
  std::size_t m = 16;
  // At most this many training rows are used (sampled with `seed`).
  std::size_t max_training_rows = 65536;
  std::size_t kmeans_iterations = 25;
  std::uint64_t seed = 42;
};

// Product quantizer (Jegou, Douze, Schmid 2011) with 8-bit codes.
//
// Distances to a query use asymmetric distance computation (ADC): the query stays exact, the
// database vector is its quantized reconstruction. For each query, a table of m x 256 partial
// distances is built once; the distance to any code is then m table lookups and adds.
//
// Metrics: L2 (sum of squared sub-distances), inner product (sum of negated sub-dot-products),
// cosine (vectors are L2-normalized before training/encoding; distance = 1 - dot).
//
// Thread safety: immutable after train(); all const methods are safe to call concurrently.
class ProductQuantizer {
 public:
  static constexpr std::size_t kCentroids = 256;

  // Fails if dim % m != 0, m == 0, or there are fewer than 256 training rows.
  [[nodiscard]] static Expected<ProductQuantizer> train(MatrixView<const float> data, Metric metric,
                                                        const PqParams& params,
                                                        ThreadPool* pool = nullptr);

  [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
  [[nodiscard]] std::size_t m() const noexcept { return m_; }
  [[nodiscard]] std::size_t sub_dim() const noexcept { return dim_ / m_; }
  [[nodiscard]] Metric metric() const noexcept { return metric_; }
  [[nodiscard]] std::size_t code_size() const noexcept { return m_; }

  // Precondition for all below: vector/query size == dim(), code size == code_size().
  void encode(std::span<const float> vector, std::span<std::uint8_t> code) const;
  // Reconstruction (for cosine, of the normalized vector).
  void decode(std::span<const std::uint8_t> code, std::span<float> out) const;

  // m x 256 table of partial distances from `query` to every centroid.
  void compute_table(std::span<const float> query, std::span<float> table) const;
  // ADC distance from the table's query to a code (same scale as the index's metric).
  [[nodiscard]] float table_distance(std::span<const float> table,
                                     std::span<const std::uint8_t> code) const noexcept;

  // Codebooks, m blocks of 256 x sub_dim, for inspection and persistence.
  [[nodiscard]] const Matrix<float>& codebooks() const noexcept { return codebooks_; }

 private:
  ProductQuantizer(std::size_t dim, std::size_t m, Metric metric, Matrix<float> codebooks)
      : dim_(dim), m_(m), metric_(metric), codebooks_(std::move(codebooks)) {}
  [[nodiscard]] std::span<const float> centroid(std::size_t sub, std::size_t c) const noexcept;

  std::size_t dim_;
  std::size_t m_;
  Metric metric_;
  Matrix<float> codebooks_;  // (m * 256) x sub_dim
};

struct PqSearchParams {
  // 0: return the top k by ADC distance (approximate distances).
  // >= k: take the top `rerank` by ADC, recompute exact distances from the stored full-precision
  // vectors, and return the best k (exact distances). Requires keep_originals.
  std::size_t rerank = 0;
};

// Flat PQ index: every vector stored as an m-byte code, searched by a full ADC scan.
//
// Memory: code_size() bytes per vector, plus 4 * dim bytes per vector when keep_originals is set
// (needed for reranking; in a deployment these would live on disk or be memory-mapped).
//
// Thread safety: concurrent const calls are safe; add_batch requires exclusive access.
class PqIndex {
 public:
  [[nodiscard]] static Expected<PqIndex> create(ProductQuantizer quantizer, bool keep_originals);

  Expected<void> add_batch(MatrixView<const float> vectors, ThreadPool* pool = nullptr);
  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::span<const float> query, std::size_t k,
                                                       const PqSearchParams& params = {}) const;

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const ProductQuantizer& quantizer() const noexcept { return pq_; }
  [[nodiscard]] std::size_t code_bytes() const noexcept { return codes_.size(); }
  [[nodiscard]] std::size_t original_bytes() const noexcept {
    return originals_.size() * sizeof(float);
  }
  [[nodiscard]] std::size_t codebook_bytes() const noexcept {
    return pq_.codebooks().data().size_bytes();
  }

 private:
  PqIndex(ProductQuantizer pq, bool keep_originals)
      : pq_(std::move(pq)), keep_originals_(keep_originals) {}

  ProductQuantizer pq_;
  bool keep_originals_;
  std::size_t size_ = 0;
  std::vector<std::uint8_t> codes_;  // size_ x m, contiguous
  std::vector<float> originals_;     // size_ x dim when keep_originals_
};

}  // namespace strata
