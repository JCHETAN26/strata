#include "strata/bm25.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "strata/brute_force.hpp"

namespace strata {
namespace {

using Tokens = std::vector<std::string>;

Bm25Index make_index(Bm25Params params = {}) {
  auto index = Bm25Index::create(params);
  EXPECT_TRUE(index.has_value());
  return std::move(*index);
}

// Independent double-precision BM25 (Lucene idf, exact lengths).
double reference_score(const std::vector<Tokens>& docs, const Tokens& query, std::size_t d,
                       double k1, double b) {
  double total = 0;
  std::size_t non_empty = 0;
  for (const auto& doc : docs) {
    total += static_cast<double>(doc.size());
    non_empty += doc.empty() ? 0 : 1;
  }
  const double n = static_cast<double>(non_empty);
  const double avgdl = total / n;
  double score = 0;
  for (const auto& term : query) {  // repeated query terms count each time
    double df = 0;
    for (const auto& doc : docs) {
      df += std::ranges::find(doc, term) != doc.end() ? 1 : 0;
    }
    if (df == 0) {
      continue;
    }
    const double tf = static_cast<double>(std::ranges::count(docs[d], term));
    const double idf = std::log(1 + (n - df + 0.5) / (df + 0.5));
    const double dl = static_cast<double>(docs[d].size());
    score += idf * tf / (tf + k1 * (1 - b + b * dl / avgdl));
  }
  return score;
}

// --- Lucene SmallFloat length encoding ----------

TEST(LuceneNorms, SmallLengthsAreExact) {
  for (std::uint32_t i = 0; i < 24; ++i) {
    EXPECT_EQ(lucene_int_to_byte4(i), i);
    EXPECT_EQ(lucene_byte4_to_int(static_cast<std::uint8_t>(i)), i);
  }
}

TEST(LuceneNorms, LongLengthsKeepFourSignificantBits) {
  EXPECT_EQ(lucene_byte4_to_int(lucene_int_to_byte4(100)), 96U);
  std::uint32_t previous = 0;
  for (std::uint32_t len = 0; len < 100000; len += 7) {
    const std::uint32_t decoded = lucene_byte4_to_int(lucene_int_to_byte4(len));
    EXPECT_LE(decoded, len);
    EXPECT_GE(decoded, previous);  // monotonic
    EXPECT_GE(static_cast<double>(decoded), 0.875 * len - 1);
    previous = decoded;
  }
  // Every byte decodes to a length that re-encodes to the same byte.
  for (int byte = 0; byte < 256; ++byte) {
    const auto decoded = lucene_byte4_to_int(static_cast<std::uint8_t>(byte));
    EXPECT_EQ(lucene_int_to_byte4(decoded), byte);
  }
}

// --- Scoring ----------

TEST(Bm25, RejectsInvalidParameters) {
  EXPECT_FALSE(Bm25Index::create({.k1 = -1.0F}).has_value());
  EXPECT_FALSE(Bm25Index::create({.b = 1.5F}).has_value());
  EXPECT_TRUE(Bm25Index::create({.k1 = 0.0F, .b = 0.0F}).has_value());
}

TEST(Bm25, MatchesReferenceFormula) {
  const std::vector<Tokens> docs = {{"apple", "banana", "apple"},
                                    {"banana", "cherry"},
                                    {"cherry", "cherry", "cherry", "date", "apple"},
                                    {"elder"}};
  for (auto encoding : {LengthEncoding::kLucene, LengthEncoding::kExact}) {
    auto index = make_index({.length_encoding = encoding});  // lengths < 24: encodings agree
    for (const auto& d : docs) {
      ASSERT_TRUE(index.add_tokens(d));
    }
    for (const Tokens& query :
         {Tokens{"apple"}, Tokens{"cherry", "banana"}, Tokens{"apple", "date"}}) {
      auto result = index.search_tokens(query, 10);
      ASSERT_TRUE(result);
      for (const auto& n : *result) {
        EXPECT_NEAR(Bm25Index::score(n), reference_score(docs, query, n.id, 0.9, 0.4), 1e-5);
      }
    }
  }
}

TEST(Bm25, RepeatedQueryTermsCountEachTime) {
  auto index = make_index();
  ASSERT_TRUE(index.add_tokens(Tokens{"x", "y"}));
  ASSERT_TRUE(index.add_tokens(Tokens{"y"}));
  const float once = Bm25Index::score(index.search_tokens(Tokens{"x"}, 1)->front());
  const float twice = Bm25Index::score(index.search_tokens(Tokens{"x", "x"}, 1)->front());
  EXPECT_FLOAT_EQ(twice, 2 * once);  // Anserini: boost = query term count
}

TEST(Bm25, ResultsSortedByScoreThenId) {
  auto index = make_index();
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(index.add_tokens(Tokens{"same", "words"}));  // identical docs tie
  }
  ASSERT_TRUE(index.add_tokens(Tokens{"same", "same", "words"}));
  auto result = index.search_tokens(Tokens{"same"}, 10);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 6U);
  EXPECT_EQ((*result)[0].id, 5U);  // higher tf wins
  for (std::size_t i = 1; i < 6; ++i) {
    EXPECT_EQ((*result)[i].id, i - 1);  // ties: ascending id, as Lucene
  }
}

TEST(Bm25, EdgeCases) {
  auto index = make_index();
  EXPECT_TRUE(index.search("anything", 10)->empty());  // empty index
  ASSERT_EQ(index.add_tokens(Tokens{}), VectorId{0});  // empty document
  ASSERT_EQ(index.add_tokens(Tokens{"word"}), VectorId{1});
  EXPECT_EQ(index.size(), 2U);
  EXPECT_EQ(index.doc_count(), 1U);                         // empty docs don't count, as in Lucene
  EXPECT_TRUE(index.search_tokens(Tokens{}, 10)->empty());  // empty query
  EXPECT_TRUE(index.search_tokens(Tokens{"unknown"}, 10)->empty());  // unknown term
  EXPECT_TRUE(index.search_tokens(Tokens{"word"}, 0)->empty());      // k = 0
  auto result = index.search_tokens(Tokens{"word", "unknown"}, 10);  // unknown terms ignored
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ((*result)[0].id, 1U);
}

TEST(Bm25, RemoveTombstonesWithoutChangingStats) {
  auto index = make_index();
  ASSERT_TRUE(index.add_tokens(Tokens{"a1", "shared"}));
  ASSERT_TRUE(index.add_tokens(Tokens{"b1", "shared"}));
  const float before = Bm25Index::score(index.search_tokens(Tokens{"shared"}, 2)->at(1));
  ASSERT_TRUE(index.remove(0));
  EXPECT_FALSE(index.remove(0));
  EXPECT_FALSE(index.remove(7));
  auto result = index.search_tokens(Tokens{"shared"}, 10);
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ((*result)[0].id, 1U);
  EXPECT_FLOAT_EQ(Bm25Index::score((*result)[0]), before);  // df and avgdl still include doc 0
  EXPECT_EQ(index.add_tokens(Tokens{"c"}), VectorId{2});    // ids are never reused
}

TEST(Bm25, LengthEncodingMattersForLongDocuments) {
  Tokens long_doc(100, "filler");
  long_doc.push_back("needle");
  Bm25Index lucene = make_index({.length_encoding = LengthEncoding::kLucene});
  Bm25Index exact = make_index({.length_encoding = LengthEncoding::kExact});
  for (Bm25Index* index : {&lucene, &exact}) {
    ASSERT_TRUE(index->add_tokens(long_doc));
    ASSERT_TRUE(index->add_tokens(Tokens{"other", "needle"}));
  }
  const float s_lucene = Bm25Index::score(lucene.search_tokens(Tokens{"needle"}, 2)->at(1));
  const float s_exact = Bm25Index::score(exact.search_tokens(Tokens{"needle"}, 2)->at(1));
  EXPECT_GT(s_lucene, s_exact);  // Lucene sees length 96, not 101: slightly less penalty
}

TEST(Bm25, AnalyzerIsAppliedToDocumentsAndQueries) {
  auto index = make_index();  // anserini_english by default
  ASSERT_TRUE(index.add("The runner was running quickly"));
  ASSERT_TRUE(index.add("Completely unrelated text"));
  auto result = index.search("RUNS", 10);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 1U);
  EXPECT_EQ((*result)[0].id, 0U);
  EXPECT_EQ(index.doc_length(0), 3U);  // runner, run, quickli ("the", "was" are stopwords)
  EXPECT_EQ(index.doc_freq("run"), 1U);
}

// BM25 and vector indexes share one id space: adding document i to both in the same order gives
// the same id, and removals keep them aligned.
TEST(Bm25, SharesIdSpaceWithVectorIndexes) {
  auto bm25 = make_index();
  auto vectors = BruteForceIndex::create(2, Metric::kL2);
  ASSERT_TRUE(vectors);
  const std::vector<std::pair<std::string, std::vector<float>>> docs = {
      {"first doc", {0, 0}}, {"second doc", {1, 0}}, {"third doc", {0, 1}}};
  for (const auto& [text, vec] : docs) {
    EXPECT_EQ(*bm25.add(text), *vectors->add(vec));
  }
  ASSERT_TRUE(bm25.remove(1));
  ASSERT_TRUE(vectors->remove(1));
  EXPECT_EQ(bm25.live_size(), vectors->live_size());
  EXPECT_EQ(*bm25.add("fourth doc"), *vectors->add(std::vector<float>{1, 1}));
}

}  // namespace
}  // namespace strata
