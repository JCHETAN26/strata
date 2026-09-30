#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "strata/bitset.hpp"
#include "strata/error.hpp"
#include "strata/types.hpp"

namespace strata {

// A metadata value: integer (years, prices, timestamps) or category (tags, sources).
using AttributeValue = std::variant<std::int64_t, std::string>;

enum class ColumnType {
  kInt,
  kCategory,
};

struct ColumnSpec {
  std::string name;
  ColumnType type;
};

// Columnar metadata, one row per vector id (row i describes VectorId i).
// Categories are interned: each distinct string gets a dense uint32 code per column, so
// filters compare integers.
//
// Thread safety: concurrent const access is safe; append requires exclusive access.
class AttributeTable {
 public:
  // Fails on an empty or duplicate column name.
  [[nodiscard]] static Expected<AttributeTable> create(std::vector<ColumnSpec> schema);

  // Appends the row for the next id. Values are in schema order and must match column types.
  Expected<void> append(const std::vector<AttributeValue>& row);

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] const std::vector<ColumnSpec>& schema() const noexcept { return schema_; }

 private:
  friend class CompiledFilter;
  struct Column {
    ColumnSpec spec;
    std::vector<std::int64_t> ints;                             // kInt
    std::vector<std::uint32_t> codes;                           // kCategory
    std::unordered_map<std::string, std::uint32_t> dictionary;  // kCategory
  };
  explicit AttributeTable(std::vector<ColumnSpec> schema);
  [[nodiscard]] const Column* find(const std::string& name) const;

  std::vector<ColumnSpec> schema_;
  std::vector<Column> columns_;
  std::size_t rows_ = 0;
};

// Filter expression, built with the factory functions and combined with all_of / any_of / negate.
// Immutable and cheap to copy (shared tree).
class Filter {
 public:
  static Filter equals(std::string column, AttributeValue value);
  // Inclusive range on an int column.
  static Filter range(std::string column, std::int64_t lo, std::int64_t hi);
  static Filter in(std::string column, std::vector<AttributeValue> values);
  static Filter all_of(std::vector<Filter> filters);
  static Filter any_of(std::vector<Filter> filters);
  static Filter negate(Filter filter);

  struct Node;
  [[nodiscard]] const Node& root() const noexcept { return *root_; }

 private:
  explicit Filter(std::shared_ptr<const Node> root) : root_(std::move(root)) {}
  std::shared_ptr<const Node> root_;
};

// A filter resolved against one AttributeTable: column names and category strings are looked up
// once, so matches() only compares integers. An unknown category value matches nothing (it
// cannot occur in the data); an unknown column or a type mismatch is a compile error.
//
// The table must outlive the CompiledFilter and must not be appended to while it is in use.
// Thread safety: immutable apart from the selectivity cache, which is filled once under
// std::call_once; safe to use from many threads. Copies share the cache (they are the same filter).
class CompiledFilter {
 public:
  [[nodiscard]] static Expected<CompiledFilter> compile(const Filter& filter,
                                                        const AttributeTable& table);

  [[nodiscard]] bool matches(VectorId id) const noexcept;
  // Number of ids the filter covers (the table's rows when compiled).
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  // Bitset of every matching id in [0, table.rows()).
  [[nodiscard]] Bitset evaluate() const;
  // Fraction of ids that match, estimated from `samples` random ids (exact if samples >= rows).
  [[nodiscard]] double estimate_selectivity(std::size_t samples, std::uint64_t seed = 1) const;
  // The same estimates, computed at most once per filter and then reused: a coarse one (1000 ids,
  // seed 1) and a precise one (20000 ids, seed 2; exact when rows <= 20000). HnswIndex's auto
  // strategy uses these, so a filter reused across queries pays for sampling once, not per query.
  static constexpr std::size_t kCoarseSamples = 1000;
  static constexpr std::size_t kPreciseSamples = 20000;
  [[nodiscard]] double coarse_selectivity() const;
  [[nodiscard]] double precise_selectivity() const;

  struct Op;

 private:
  CompiledFilter(std::vector<Op> ops, std::size_t rows);
  [[nodiscard]] bool eval(std::size_t op, VectorId id) const noexcept;

  // Flattened expression tree; ops_[0] is the root.
  std::vector<Op> ops_;
  std::size_t rows_;

  struct SelectivityCache {
    std::once_flag coarse_once;
    std::once_flag precise_once;
    double coarse = 0;
    double precise = 0;
  };
  // Shared, so the filter stays copyable and copies reuse each other's estimates.
  std::shared_ptr<SelectivityCache> cache_;
};

// Opcode for one node of a compiled filter. Public only so the implementation can build it.
struct CompiledFilter::Op {
  enum class Kind { kIntRange, kIntIn, kCodeIn, kAll, kAny, kNot, kNever } kind;
  const std::vector<std::int64_t>* ints = nullptr;
  const std::vector<std::uint32_t>* codes = nullptr;
  std::int64_t lo = 0;
  std::int64_t hi = 0;
  std::vector<std::uint32_t> code_set;  // sorted, distinct; kCodeIn
  std::vector<std::int64_t> int_set;    // sorted, distinct; kIntIn
  std::vector<std::size_t> children;    // kAll, kAny, kNot
};

}  // namespace strata
