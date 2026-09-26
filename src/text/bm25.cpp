#include "strata/bm25.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <string>

namespace strata {

// --- Lucene SmallFloat (org.apache.lucene.util.SmallFloat) ----------

namespace {

std::uint32_t long_to_int4(std::uint64_t i) {
  const int num_bits = 64 - std::countl_zero(i);
  if (num_bits < 4) {
    return static_cast<std::uint32_t>(i);  // subnormal
  }
  const int shift = num_bits - 4;
  auto encoded = static_cast<std::uint32_t>(i >> static_cast<unsigned>(shift));
  encoded &= 0x07U;  // the most significant bit is implicit
  encoded |= static_cast<std::uint32_t>(shift + 1) << 3U;
  return encoded;
}

std::uint64_t int4_to_long(std::uint32_t i) {
  const std::uint64_t bits = i & 0x07U;
  const int shift = static_cast<int>(i >> 3U) - 1;
  return shift == -1 ? bits : (bits | 0x08U) << static_cast<unsigned>(shift);
}

const std::uint32_t kMaxInt4 = long_to_int4(std::numeric_limits<std::int32_t>::max());
const std::uint32_t kNumFreeValues = 255 - kMaxInt4;  // = 24

}  // namespace

std::uint8_t lucene_int_to_byte4(std::uint32_t value) noexcept {
  if (value < kNumFreeValues) {
    return static_cast<std::uint8_t>(value);
  }
  return static_cast<std::uint8_t>(kNumFreeValues + long_to_int4(value - kNumFreeValues));
}

std::uint32_t lucene_byte4_to_int(std::uint8_t encoded) noexcept {
  const std::uint32_t i = encoded;
  if (i < kNumFreeValues) {
    return i;
  }
  return static_cast<std::uint32_t>(kNumFreeValues + int4_to_long(i - kNumFreeValues));
}

// --- Bm25Index ----------

Expected<Bm25Index> Bm25Index::create(Bm25Params params) {
  if (!(params.k1 >= 0.0F) || !(params.b >= 0.0F && params.b <= 1.0F)) {
    return make_error(ErrorCode::kInvalidArgument, "BM25 needs k1 >= 0 and 0 <= b <= 1");
  }
  return Bm25Index(params);
}

Expected<VectorId> Bm25Index::add(std::string_view text) {
  const auto tokens = analyzer_.analyze(text);
  return add_tokens(tokens);
}

Expected<VectorId> Bm25Index::add_tokens(std::span<const std::string> tokens) {
  if (lengths_.size() >= std::numeric_limits<VectorId>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "index is full");
  }
  if (tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_error(ErrorCode::kInvalidArgument, "document too long");
  }
  const auto id = static_cast<VectorId>(lengths_.size());
  // Term frequencies for this document.
  std::unordered_map<std::string_view, std::uint32_t> tf;
  for (const auto& token : tokens) {
    ++tf[token];
  }
  for (const auto& [term, count] : tf) {
    auto [it, inserted] =
        term_ids_.try_emplace(std::string(term), static_cast<std::uint32_t>(postings_.size()));
    if (inserted) {
      postings_.emplace_back();
    }
    postings_[it->second].push_back({id, count});
  }
  const auto length = static_cast<std::uint32_t>(tokens.size());
  lengths_.push_back(length);
  norms_.push_back(lucene_int_to_byte4(length));
  deleted_.push_back(0);
  total_terms_ += length;
  non_empty_docs_ += length > 0 ? 1 : 0;
  return id;
}

Expected<void> Bm25Index::remove(VectorId id) {
  if (id >= lengths_.size() || deleted_[id] != 0) {
    return make_error(ErrorCode::kNotFound, "no live document with id " + std::to_string(id));
  }
  deleted_[id] = 1;
  ++num_deleted_;
  return {};
}

bool Bm25Index::is_deleted(VectorId id) const noexcept {
  return id < deleted_.size() && deleted_[id] != 0;
}

std::uint32_t Bm25Index::doc_length(VectorId id) const noexcept {
  return id < lengths_.size() ? lengths_[id] : 0;
}

std::size_t Bm25Index::doc_freq(std::string_view term) const {
  const auto it = term_ids_.find(std::string(term));
  return it == term_ids_.end() ? 0 : postings_[it->second].size();
}

Expected<std::vector<Neighbor>> Bm25Index::search(std::string_view query, std::size_t k) const {
  const auto tokens = analyzer_.analyze(query);
  return search_tokens(tokens, k);
}

Expected<std::vector<Neighbor>> Bm25Index::search_tokens(std::span<const std::string> tokens,
                                                         std::size_t k) const {
  std::vector<Neighbor> heap;
  if (k == 0 || non_empty_docs_ == 0 || tokens.empty()) {
    return heap;
  }

  // Unique query terms with their counts (Anserini: boost = count), in first-occurrence order.
  std::vector<std::pair<std::uint32_t, float>> terms;
  for (const auto& token : tokens) {
    const auto it = term_ids_.find(token);
    if (it == term_ids_.end()) {
      continue;
    }
    auto found = std::ranges::find(terms, it->second, &std::pair<std::uint32_t, float>::first);
    if (found != terms.end()) {
      found->second += 1.0F;
    } else {
      terms.emplace_back(it->second, 1.0F);
    }
  }
  if (terms.empty()) {
    return heap;
  }

  // Lucene float arithmetic (BM25Similarity.scorer / BM25Scorer.doScore).
  const float k1 = params_.k1;
  const float b = params_.b;
  const auto doc_count = static_cast<double>(non_empty_docs_);
  const auto avgdl = static_cast<float>(static_cast<double>(total_terms_) / doc_count);
  std::array<float, 256> norm_cache{};
  for (std::size_t i = 0; i < norm_cache.size(); ++i) {
    const auto length = static_cast<float>(lucene_byte4_to_int(static_cast<std::uint8_t>(i)));
    norm_cache[i] = 1.0F / (k1 * ((1 - b) + b * length / avgdl));
  }
  const bool lucene_norms = params_.length_encoding == LengthEncoding::kLucene;

  // Term-at-a-time accumulation (Lucene sums clause scores in double).
  std::vector<double> scores(lengths_.size(), 0.0);
  std::vector<VectorId> touched;
  for (const auto& [term, boost] : terms) {
    const auto& list = postings_[term];
    const auto df = static_cast<double>(list.size());
    const auto idf = static_cast<float>(std::log(1.0 + (doc_count - df + 0.5) / (df + 0.5)));
    const float weight = boost * idf;
    for (const Posting& p : list) {
      const float norm_inverse =
          lucene_norms ? norm_cache[norms_[p.doc]]
                       : 1.0F / (k1 * ((1 - b) + b * static_cast<float>(lengths_[p.doc]) / avgdl));
      const float term_score = weight - weight / (1.0F + static_cast<float>(p.tf) * norm_inverse);
      if (scores[p.doc] == 0.0) {
        touched.push_back(p.doc);
      }
      scores[p.doc] += term_score;
    }
  }

  // Top k by (score desc, id asc) == Neighbor order on (-score, id).
  heap.reserve(k + 1);
  for (VectorId doc : touched) {
    if (deleted_[doc] != 0) {
      continue;
    }
    const Neighbor candidate{doc, -static_cast<float>(scores[doc])};
    if (heap.size() < k) {
      heap.push_back(candidate);
      std::push_heap(heap.begin(), heap.end());
    } else if (candidate < heap.front()) {
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = candidate;
      std::push_heap(heap.begin(), heap.end());
    }
  }
  std::sort_heap(heap.begin(), heap.end());
  return heap;
}

}  // namespace strata
