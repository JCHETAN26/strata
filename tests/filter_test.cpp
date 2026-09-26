#include "strata/filter.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "strata/bitset.hpp"
#include "strata/brute_force.hpp"
#include "test_util.hpp"

namespace strata {
namespace {

// --- Bitset -------------------------------------------------------------------------------------

TEST(Bitset, SetTestCountIterate) {
  Bitset b(130);
  for (std::size_t i : {0U, 1U, 63U, 64U, 127U, 129U}) {
    b.set(i);
  }
  EXPECT_EQ(b.count(), 6U);
  EXPECT_TRUE(b.test(64));
  EXPECT_FALSE(b.test(65));
  b.reset(64);
  std::vector<std::size_t> seen;
  b.for_each_set([&](std::size_t i) { seen.push_back(i); });
  EXPECT_EQ(seen, (std::vector<std::size_t>{0, 1, 63, 127, 129}));
}

TEST(Bitset, Empty) {
  Bitset b(0);
  EXPECT_EQ(b.count(), 0U);
  int calls = 0;
  b.for_each_set([&](std::size_t) { ++calls; });
  EXPECT_EQ(calls, 0);
}

// --- Attribute table
// ------------------------------------------------------------------------------

AttributeTable make_table() {
  auto t = AttributeTable::create({{"year", ColumnType::kInt}, {"source", ColumnType::kCategory}});
  EXPECT_TRUE(t);
  const std::vector<std::pair<std::int64_t, std::string>> rows = {
      {2019, "arxiv"}, {2020, "blog"}, {2021, "arxiv"}, {2022, "news"}, {2023, "blog"}};
  for (const auto& [year, source] : rows) {
    EXPECT_TRUE(t->append({year, source}));
  }
  return std::move(*t);
}

std::vector<VectorId> matching(const Filter& f, const AttributeTable& t) {
  auto compiled = CompiledFilter::compile(f, t);
  EXPECT_TRUE(compiled) << compiled.error().message;
  std::vector<VectorId> out;
  for (VectorId id = 0; id < t.rows(); ++id) {
    if (compiled->matches(id)) {
      out.push_back(id);
    }
  }
  // evaluate() must agree with matches().
  std::vector<VectorId> from_bits;
  compiled->evaluate().for_each_set(
      [&](std::size_t i) { from_bits.push_back(static_cast<VectorId>(i)); });
  EXPECT_EQ(out, from_bits);
  return out;
}

using Ids = std::vector<VectorId>;

TEST(AttributeTable, SchemaValidation) {
  EXPECT_FALSE(AttributeTable::create({{"a", ColumnType::kInt}, {"a", ColumnType::kCategory}}));
  EXPECT_FALSE(AttributeTable::create({{"", ColumnType::kInt}}));
  auto t = AttributeTable::create({{"year", ColumnType::kInt}});
  ASSERT_TRUE(t);
  EXPECT_FALSE(t->append({std::string("2020")}));               // wrong type
  EXPECT_FALSE(t->append({std::int64_t{1}, std::int64_t{2}}));  // wrong arity
  EXPECT_EQ(t->rows(), 0U);
}

TEST(Filter, Predicates) {
  const auto t = make_table();
  EXPECT_EQ(matching(Filter::equals("source", std::string("arxiv")), t), (Ids{0, 2}));
  EXPECT_EQ(matching(Filter::equals("year", std::int64_t{2022}), t), (Ids{3}));
  EXPECT_EQ(matching(Filter::range("year", 2020, 2022), t), (Ids{1, 2, 3}));
  EXPECT_EQ(matching(Filter::in("source", {std::string("news"), std::string("blog")}), t),
            (Ids{1, 3, 4}));
  EXPECT_EQ(matching(Filter::in("year", {std::int64_t{2019}, std::int64_t{2023}}), t), (Ids{0, 4}));
}

TEST(Filter, BooleanCombinators) {
  const auto t = make_table();
  const auto arxiv = Filter::equals("source", std::string("arxiv"));
  const auto recent = Filter::range("year", 2021, 9999);
  EXPECT_EQ(matching(Filter::all_of({arxiv, recent}), t), (Ids{2}));
  EXPECT_EQ(matching(Filter::any_of({arxiv, recent}), t), (Ids{0, 2, 3, 4}));
  EXPECT_EQ(matching(Filter::negate(arxiv), t), (Ids{1, 3, 4}));
  EXPECT_EQ(matching(Filter::all_of({}), t), (Ids{0, 1, 2, 3, 4}));  // empty AND = true
  EXPECT_EQ(matching(Filter::any_of({}), t), (Ids{}));               // empty OR = false
}

TEST(Filter, UnknownCategoryMatchesNothing) {
  const auto t = make_table();
  EXPECT_EQ(matching(Filter::equals("source", std::string("podcast")), t), (Ids{}));
  EXPECT_EQ(matching(Filter::negate(Filter::equals("source", std::string("podcast"))), t),
            (Ids{0, 1, 2, 3, 4}));
}

TEST(Filter, CompileErrors) {
  const auto t = make_table();
  EXPECT_FALSE(CompiledFilter::compile(Filter::equals("nope", std::int64_t{1}), t));
  EXPECT_FALSE(CompiledFilter::compile(Filter::range("source", 0, 1), t));
  EXPECT_FALSE(CompiledFilter::compile(Filter::equals("year", std::string("2020")), t));
  EXPECT_FALSE(CompiledFilter::compile(Filter::equals("source", std::int64_t{1}), t));
  // Errors inside combinators propagate.
  EXPECT_FALSE(CompiledFilter::compile(
      Filter::all_of({Filter::range("year", 0, 1), Filter::equals("nope", std::int64_t{1})}), t));
}

TEST(Filter, OutOfRangeIdDoesNotMatch) {
  const auto t = make_table();
  auto f = CompiledFilter::compile(Filter::range("year", 0, 9999), t);
  ASSERT_TRUE(f);
  EXPECT_FALSE(f->matches(5));
}

TEST(Filter, SelectivityEstimate) {
  auto t = AttributeTable::create({{"bucket", ColumnType::kInt}});
  ASSERT_TRUE(t);
  for (std::int64_t i = 0; i < 10000; ++i) {
    ASSERT_TRUE(t->append({i % 100}));
  }
  auto f = CompiledFilter::compile(Filter::range("bucket", 0, 9), *t);  // exactly 10%
  ASSERT_TRUE(f);
  EXPECT_DOUBLE_EQ(f->estimate_selectivity(1'000'000), 0.10);  // exact path
  EXPECT_NEAR(f->estimate_selectivity(2000), 0.10, 0.03);      // sampled
}

// --- Filtered brute-force search -----------------------------------------------------------------

TEST(FilteredSearch, PrefilterAndPredicateMatchFilteredExactSearch) {
  const auto base = test::random_matrix(3000, 16, 1);
  const auto queries = test::random_matrix(30, 16, 2);
  auto index = BruteForceIndex::create(16, Metric::kL2);
  ASSERT_TRUE(index && index->add_batch(base));
  auto table = AttributeTable::create({{"bucket", ColumnType::kInt}});
  ASSERT_TRUE(table);
  std::mt19937 rng(3);
  for (std::size_t i = 0; i < base.rows(); ++i) {
    ASSERT_TRUE(table->append({static_cast<std::int64_t>(rng() % 100)}));
  }
  for (std::int64_t upper : {0, 9, 49, 99}) {  // ~1%, 10%, 50%, 100%
    auto f = CompiledFilter::compile(Filter::range("bucket", 0, upper), *table);
    ASSERT_TRUE(f);
    const Bitset allowed = f->evaluate();
    for (std::size_t q = 0; q < queries.rows(); ++q) {
      // Reference: exact distances over matching ids, sorted.
      std::vector<Neighbor> expected;
      allowed.for_each_set([&](std::size_t i) {
        expected.push_back(
            {static_cast<VectorId>(i), distance(Metric::kL2, queries.row(q), base.row(i))});
      });
      std::ranges::sort(expected);
      expected.resize(std::min<std::size_t>(10, expected.size()));

      auto pre = index->search_filtered(queries.row(q), 10, allowed);
      auto pred =
          index->search_predicate(queries.row(q), 10, [&](VectorId id) { return f->matches(id); });
      ASSERT_TRUE(pre && pred);
      EXPECT_EQ(*pre, expected) << "upper " << upper;
      EXPECT_EQ(*pred, expected) << "upper " << upper;
    }
  }
}

TEST(FilteredSearch, RespectsTombstonesAndEdgeCases) {
  auto index = BruteForceIndex::create(1, Metric::kL2);
  ASSERT_TRUE(index);
  for (float x : {0.0F, 1.0F, 2.0F, 3.0F}) {
    ASSERT_TRUE(index->add(std::vector<float>{x}));
  }
  ASSERT_TRUE(index->remove(1));
  Bitset allowed(4);
  allowed.set(1);
  allowed.set(3);
  auto r = index->search_filtered(std::vector<float>{0}, 10, allowed);
  ASSERT_TRUE(r);
  ASSERT_EQ(r->size(), 1U);
  EXPECT_EQ((*r)[0].id, 3U);
  EXPECT_TRUE(index->search_filtered(std::vector<float>{0}, 10, Bitset(4))->empty());
  EXPECT_FALSE(index->search_filtered(std::vector<float>{0}, 10, Bitset(5)));   // wrong size
  EXPECT_FALSE(index->search_filtered(std::vector<float>{0, 1}, 10, allowed));  // wrong dim
  EXPECT_TRUE(index->search_filtered(std::vector<float>{0}, 0, allowed)->empty());
}

}  // namespace
}  // namespace strata
