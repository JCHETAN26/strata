#include "strata/filter.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <set>

namespace strata {

// --- AttributeTable ----------------------------------------------------------------------------

AttributeTable::AttributeTable(std::vector<ColumnSpec> schema) : schema_(std::move(schema)) {
  for (const auto& spec : schema_) {
    columns_.push_back(Column{.spec = spec, .ints = {}, .codes = {}, .dictionary = {}});
  }
}

Expected<AttributeTable> AttributeTable::create(std::vector<ColumnSpec> schema) {
  std::set<std::string> names;
  for (const auto& spec : schema) {
    if (spec.name.empty() || !names.insert(spec.name).second) {
      return make_error(ErrorCode::kInvalidArgument,
                        "column names must be non-empty and unique: '" + spec.name + "'");
    }
  }
  return AttributeTable(std::move(schema));
}

Expected<void> AttributeTable::append(const std::vector<AttributeValue>& row) {
  if (row.size() != columns_.size()) {
    return make_error(ErrorCode::kInvalidArgument,
                      "row has " + std::to_string(row.size()) + " values for " +
                          std::to_string(columns_.size()) + " columns");
  }
  for (std::size_t c = 0; c < columns_.size(); ++c) {
    const bool is_int = std::holds_alternative<std::int64_t>(row[c]);
    if (is_int != (columns_[c].spec.type == ColumnType::kInt)) {
      return make_error(ErrorCode::kInvalidArgument,
                        "wrong value type for column '" + columns_[c].spec.name + "'");
    }
  }
  for (std::size_t c = 0; c < columns_.size(); ++c) {
    auto& column = columns_[c];
    if (column.spec.type == ColumnType::kInt) {
      column.ints.push_back(std::get<std::int64_t>(row[c]));
    } else {
      const auto& value = std::get<std::string>(row[c]);
      const auto next = static_cast<std::uint32_t>(column.dictionary.size());
      const auto [it, inserted] = column.dictionary.try_emplace(value, next);
      column.codes.push_back(it->second);
    }
  }
  ++rows_;
  return {};
}

const AttributeTable::Column* AttributeTable::find(const std::string& name) const {
  for (const auto& column : columns_) {
    if (column.spec.name == name) {
      return &column;
    }
  }
  return nullptr;
}

// --- Filter ------------------------------------------------------------------------------------

struct Filter::Node {
  enum class Kind { kEquals, kRange, kIn, kAll, kAny, kNot } kind;
  std::string column;
  std::vector<AttributeValue> values;  // kEquals (one), kIn
  std::int64_t lo = 0;
  std::int64_t hi = 0;
  std::vector<Filter> children;
};

Filter Filter::equals(std::string column, AttributeValue value) {
  return Filter(std::make_shared<const Node>(Node{
      .kind = Node::Kind::kEquals, .column = std::move(column), .values = {std::move(value)}}));
}

Filter Filter::range(std::string column, std::int64_t lo, std::int64_t hi) {
  return Filter(std::make_shared<const Node>(
      Node{.kind = Node::Kind::kRange, .column = std::move(column), .lo = lo, .hi = hi}));
}

Filter Filter::in(std::string column, std::vector<AttributeValue> values) {
  return Filter(std::make_shared<const Node>(
      Node{.kind = Node::Kind::kIn, .column = std::move(column), .values = std::move(values)}));
}

Filter Filter::all_of(std::vector<Filter> filters) {
  return Filter(
      std::make_shared<const Node>(Node{.kind = Node::Kind::kAll, .children = std::move(filters)}));
}

Filter Filter::any_of(std::vector<Filter> filters) {
  return Filter(
      std::make_shared<const Node>(Node{.kind = Node::Kind::kAny, .children = std::move(filters)}));
}

Filter Filter::negate(Filter filter) {
  return Filter(std::make_shared<const Node>(
      Node{.kind = Node::Kind::kNot, .children = {std::move(filter)}}));
}

// --- CompiledFilter ----------------------------------------------------------------------------

namespace {

using Op = CompiledFilter::Op;
using Node = Filter::Node;

tl::unexpected<Error> type_error(const std::string& column, const char* expected) {
  return make_error(ErrorCode::kInvalidArgument,
                    "column '" + column + "' is not " + expected + " for this predicate");
}

}  // namespace

CompiledFilter::CompiledFilter(std::vector<Op> ops, std::size_t rows)
    : ops_(std::move(ops)), rows_(rows) {}

Expected<CompiledFilter> CompiledFilter::compile(const Filter& filter,
                                                 const AttributeTable& table) {
  std::vector<Op> ops;
  // Recursive lambda over the tree. ops[0] must be the root, so reserve it first.
  std::function<Expected<std::size_t>(const Node&)> build =
      [&](const Node& node) -> Expected<std::size_t> {
    const std::size_t index = ops.size();
    ops.push_back(Op{.kind = Op::Kind::kNever});
    Op op{.kind = Op::Kind::kNever};
    switch (node.kind) {
      case Node::Kind::kAll:
      case Node::Kind::kAny:
      case Node::Kind::kNot: {
        op.kind = node.kind == Node::Kind::kAll   ? Op::Kind::kAll
                  : node.kind == Node::Kind::kAny ? Op::Kind::kAny
                                                  : Op::Kind::kNot;
        for (const auto& child : node.children) {
          auto c = build(child.root());
          if (!c) {
            return c;
          }
          op.children.push_back(*c);
        }
        break;
      }
      case Node::Kind::kRange:
      case Node::Kind::kEquals:
      case Node::Kind::kIn: {
        const auto* column = table.find(node.column);
        if (column == nullptr) {
          return make_error(ErrorCode::kInvalidArgument, "unknown column '" + node.column + "'");
        }
        if (node.kind == Node::Kind::kRange) {
          if (column->spec.type != ColumnType::kInt) {
            return type_error(node.column, "an int column");
          }
          op = Op{.kind = Op::Kind::kIntRange, .ints = &column->ints, .lo = node.lo, .hi = node.hi};
          break;
        }
        if (column->spec.type == ColumnType::kInt) {
          // Equality / membership on ints: an OR of single-point ranges.
          op.kind = Op::Kind::kAny;
          for (const auto& value : node.values) {
            if (!std::holds_alternative<std::int64_t>(value)) {
              return type_error(node.column, "a category column");
            }
            const auto v = std::get<std::int64_t>(value);
            op.children.push_back(ops.size());
            ops.push_back(Op{.kind = Op::Kind::kIntRange, .ints = &column->ints, .lo = v, .hi = v});
          }
        } else {
          op.kind = Op::Kind::kCodeIn;
          op.codes = &column->codes;
          for (const auto& value : node.values) {
            if (!std::holds_alternative<std::string>(value)) {
              return type_error(node.column, "an int column");
            }
            const auto it = column->dictionary.find(std::get<std::string>(value));
            if (it != column->dictionary.end()) {
              op.code_set.push_back(it->second);  // unknown strings can never match
            }
          }
          std::ranges::sort(op.code_set);
          if (op.code_set.empty()) {
            op.kind = Op::Kind::kNever;
          }
        }
        break;
      }
    }
    ops[index] = std::move(op);
    return index;
  };
  auto root = build(filter.root());
  if (!root) {
    return tl::unexpected(root.error());
  }
  return CompiledFilter(std::move(ops), table.rows());
}

bool CompiledFilter::eval(std::size_t index, VectorId id) const noexcept {
  const Op& op = ops_[index];
  switch (op.kind) {
    case Op::Kind::kIntRange: {
      const std::int64_t v = (*op.ints)[id];
      return v >= op.lo && v <= op.hi;
    }
    case Op::Kind::kCodeIn: {
      const std::uint32_t code = (*op.codes)[id];
      // Small sets (the common case) are faster to scan than to binary-search.
      return std::ranges::find(op.code_set, code) != op.code_set.end();
    }
    case Op::Kind::kAll:
      return std::ranges::all_of(op.children, [&](std::size_t c) { return eval(c, id); });
    case Op::Kind::kAny:
      return std::ranges::any_of(op.children, [&](std::size_t c) { return eval(c, id); });
    case Op::Kind::kNot:
      return !eval(op.children[0], id);
    case Op::Kind::kNever:
      return false;
  }
  return false;
}

bool CompiledFilter::matches(VectorId id) const noexcept { return id < rows_ && eval(0, id); }

Bitset CompiledFilter::evaluate() const {
  Bitset bits(rows_);
  for (std::size_t i = 0; i < rows_; ++i) {
    if (eval(0, static_cast<VectorId>(i))) {
      bits.set(i);
    }
  }
  return bits;
}

double CompiledFilter::estimate_selectivity(std::size_t samples, std::uint64_t seed) const {
  if (rows_ == 0) {
    return 0.0;
  }
  if (samples >= rows_) {
    return static_cast<double>(evaluate().count()) / static_cast<double>(rows_);
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, rows_ - 1);
  std::size_t hits = 0;
  for (std::size_t s = 0; s < samples; ++s) {
    hits += eval(0, static_cast<VectorId>(pick(rng))) ? 1 : 0;
  }
  return static_cast<double>(hits) / static_cast<double>(samples);
}

}  // namespace strata
