#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "strata/error.hpp"
#include "strata/text.hpp"
#include "strata/types.hpp"

namespace strata {

// How document lengths enter the BM25 length normalization.
enum class LengthEncoding {
  // Lucene's lossy 1-byte norm (SmallFloat.intToByte4 / byte4ToInt): lengths up to 23 are exact,
  // longer ones keep 4 significant bits. Needed to reproduce Lucene/Anserini scores exactly.
  kLucene,
  // Exact token counts (what bm25s and most textbook implementations use).
  kExact,
};

struct Bm25Params {
  float k1 = 0.9F;  // Anserini's BEIR defaults
  float b = 0.4F;
  LengthEncoding length_encoding = LengthEncoding::kLucene;
  AnalyzerConfig analyzer = AnalyzerConfig::anserini_english();
};

// Okapi BM25 over an inverted index, scored exactly as Lucene's BM25Similarity:
//
//   idf(t)    = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))              (Lucene / Anserini variant)
//   score(d)  = sum over query terms t of  qtf(t) * idf(t) * tf / (tf + k1 * (1 - b + b * dl /
//   avgdl))
//
// N counts documents with at least one token, avgdl = total tokens / N, and a query term that
// appears qtf times counts qtf times (Anserini turns repeats into a boost of qtf). Arithmetic
// follows Lucene's float operations (term scores computed as w - w / (1 + tf * normInverse),
// summed in double), so scores match Lucene's to float precision.
//
// Document ids are dense and assigned in insertion order, the same id space as the vector indexes
// (BruteForceIndex, PqIndex, HnswIndex): add document i to every index in the same order and the
// ids line up, so hybrid retrieval can fuse results by id. remove() tombstones an id (it is never
// reused), keeping the spaces aligned. Like Lucene before segment merges, deleted documents still
// count in N, df, and avgdl.
//
// search() returns Neighbors sorted by descending score, ties by ascending id (Lucene's order).
// For uniformity with the vector indexes, Neighbor::distance holds the *negated* score (lower is
// better); use score() or -distance.
//
// Thread safety: concurrent const calls (search, stats) are safe; add/remove need exclusive access.
class Bm25Index {
 public:
  // Fails if k1 < 0 or b is outside [0, 1].
  [[nodiscard]] static Expected<Bm25Index> create(Bm25Params params = {});

  // Analyzes and indexes one document; returns its id.
  Expected<VectorId> add(std::string_view text);
  // Indexes pre-analyzed tokens (bypasses the analyzer); returns the id.
  Expected<VectorId> add_tokens(std::span<const std::string> tokens);
  // Tombstones a document. kNotFound if id >= size() or already removed.
  Expected<void> remove(VectorId id);

  [[nodiscard]] Expected<std::vector<Neighbor>> search(std::string_view query, std::size_t k) const;
  [[nodiscard]] Expected<std::vector<Neighbor>> search_tokens(std::span<const std::string> tokens,
                                                              std::size_t k) const;

  [[nodiscard]] static float score(const Neighbor& n) noexcept { return -n.distance; }

  [[nodiscard]] std::size_t size() const noexcept { return lengths_.size(); }
  [[nodiscard]] std::size_t live_size() const noexcept { return lengths_.size() - num_deleted_; }
  [[nodiscard]] bool is_deleted(VectorId id) const noexcept;
  [[nodiscard]] std::size_t vocabulary_size() const noexcept { return postings_.size(); }
  // Sum of document lengths in tokens (Lucene's sumTotalTermFreq).
  [[nodiscard]] std::uint64_t total_terms() const noexcept { return total_terms_; }
  // Documents with at least one token (Lucene's docCount).
  [[nodiscard]] std::size_t doc_count() const noexcept { return non_empty_docs_; }
  [[nodiscard]] std::uint32_t doc_length(VectorId id) const noexcept;
  [[nodiscard]] std::size_t doc_freq(std::string_view term) const;
  [[nodiscard]] const Analyzer& analyzer() const noexcept { return analyzer_; }
  [[nodiscard]] const Bm25Params& params() const noexcept { return params_; }

 private:
  struct Posting {
    VectorId doc;
    std::uint32_t tf;
  };

  explicit Bm25Index(Bm25Params params) : params_(params), analyzer_(params.analyzer) {}

  Bm25Params params_;
  Analyzer analyzer_;
  std::unordered_map<std::string, std::uint32_t> term_ids_;
  std::vector<std::vector<Posting>> postings_;  // per term, sorted by doc id (insertion order)
  std::vector<std::uint32_t> lengths_;          // tokens per document
  std::vector<std::uint8_t> norms_;             // Lucene-encoded lengths
  std::vector<std::uint8_t> deleted_;
  std::size_t num_deleted_ = 0;
  std::size_t non_empty_docs_ = 0;
  std::uint64_t total_terms_ = 0;
};

// Lucene SmallFloat length encoding (exposed for tests).
[[nodiscard]] std::uint8_t lucene_int_to_byte4(std::uint32_t value) noexcept;
[[nodiscard]] std::uint32_t lucene_byte4_to_int(std::uint8_t encoded) noexcept;

}  // namespace strata
