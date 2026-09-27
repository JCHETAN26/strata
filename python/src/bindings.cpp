// Python bindings for Strata (module strata._core), built with nanobind.
//
// Conventions (also in the module docstring):
// - Inputs: float32 C-contiguous NumPy arrays are passed to C++ as views, without copying. Other
//   dtypes or layouts (float64, Fortran order, slices) are converted to float32 C order with one
//   copy by nanobind's implicit conversion.
// - Outputs: result buffers are allocated in C++ and handed to NumPy (no copy).
// - Search results: (ids, distances) arrays of shape (k,) or (num_queries, k); rows with fewer than
//   k results are padded with id -1 and distance +inf (the FAISS convention).
// - Concurrency: searches, batch operations, and training release the GIL. Each index holds a
//   reader/writer lock: searches share it and run in parallel; add/remove take it exclusively,
//   so inserts from Python threads are serialized. The GIL is always released *before* the index
//   lock is taken; taking them in the other order can deadlock (a thread holding the GIL waits for
//   the lock while the lock holder waits for the GIL).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "strata/bitset.hpp"
#include "strata/bm25.hpp"
#include "strata/brute_force.hpp"
#include "strata/build_info.hpp"
#include "strata/distance.hpp"
#include "strata/filter.hpp"
#include "strata/fusion.hpp"
#include "strata/pq.hpp"
#include "strata/text.hpp"
#include "strata/thread_pool.hpp"
#ifdef STRATA_HAS_HNSW
#include "strata/hnsw.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// --- Errors ----------

[[noreturn]] void raise(const strata::Error& error) {
  switch (error.code) {
    case strata::ErrorCode::kInvalidArgument:
    case strata::ErrorCode::kDimensionMismatch:
      throw nb::value_error(error.message.c_str());
    case strata::ErrorCode::kNotFound:
      throw nb::key_error(error.message.c_str());
    case strata::ErrorCode::kIoError:
    case strata::ErrorCode::kCorruptData:
      break;
  }
  throw std::runtime_error(error.message);
}

template <typename T>
T unwrap(strata::Expected<T>&& result) {
  if (!result) {
    raise(result.error());
  }
  if constexpr (!std::is_void_v<T>) {
    return std::move(*result);
  }
}

strata::Metric parse_metric(const std::string& name) {
  const auto metric = strata::parse_metric(name);
  if (!metric) {
    throw nb::value_error(("unknown metric '" + name + "' (expected l2, ip, cosine)").c_str());
  }
  return *metric;
}

strata::KernelSet parse_kernel(const std::string& name) {
  if (name == "best") {
    return strata::KernelSet::kBest;
  }
  if (name == "scalar") {
    return strata::KernelSet::kScalar;
  }
  throw nb::value_error(("unknown kernel '" + name + "' (expected best, scalar)").c_str());
}

// --- Arrays ----------

// Any-rank float32 C-contiguous input; implicit conversion (with a copy) from other dtypes/layouts.
using FloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using FloatVector = nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using U8Array = nb::ndarray<const std::uint8_t, nb::c_contig, nb::device::cpu>;
using BoolMask = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IdArray = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// A 1-D (one row) or 2-D input viewed as rows.
struct Rows {
  strata::MatrixView<const float> view;
  bool single;  // input was 1-D: return 1-D results
};

Rows as_rows(const FloatArray& a, const char* what) {
  if (a.ndim() == 1) {
    return {{a.data(), 1, a.shape(0)}, true};
  }
  if (a.ndim() == 2) {
    return {{a.data(), a.shape(0), a.shape(1)}, false};
  }
  throw nb::value_error((std::string(what) + " must be a 1-D or 2-D array").c_str());
}

// Hands a C++ vector to NumPy without copying: the vector moves to the heap and a capsule owns it.
template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values, std::vector<std::size_t> shape) {
  auto owned = std::make_unique<std::vector<T>>(std::move(values));
  T* data = owned->data();
  // Ownership passes to the capsule; its deleter is the only place this buffer is freed.
  nb::capsule owner(owned.get(), [](void* p) noexcept {
    std::unique_ptr<std::vector<T>>(static_cast<std::vector<T>*>(p));
  });
  owned.release();
  return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
}

// Bool mask for NumPy: one byte per entry (0/1), exposed with dtype bool.
nb::ndarray<nb::numpy, bool> to_numpy_bool(std::vector<std::uint8_t>&& values) {
  const std::size_t n = values.size();
  auto owned = std::make_unique<std::vector<std::uint8_t>>(std::move(values));
  auto* data = reinterpret_cast<bool*>(owned->data());
  nb::capsule owner(owned.get(), [](void* p) noexcept {
    std::unique_ptr<std::vector<std::uint8_t>>(static_cast<std::vector<std::uint8_t>*>(p));
  });
  owned.release();
  return nb::ndarray<nb::numpy, bool>(data, 1, &n, owner);
}

// Runs fn() with the GIL released and the index lock held (shared or exclusive), in that order.
template <typename Lock, typename Mutex, typename Fn>
auto locked(Mutex& mutex, const Fn& fn) {
  nb::gil_scoped_release release;
  const Lock lock(mutex);
  return fn();
}
template <typename Mutex, typename Fn>
auto shared(Mutex& mutex, const Fn& fn) {
  return locked<std::shared_lock<std::shared_mutex>>(mutex, fn);
}

using SearchResult = std::pair<nb::ndarray<nb::numpy, std::int64_t>, nb::ndarray<nb::numpy, float>>;

// Padded (ids, distances) buffers for `rows` queries of k results each.
// For vector indexes the values are distances (padding +inf). For BM25 they are scores: C++
// returns negated scores in Neighbor::distance, so `negate` flips them back (padding -inf).
struct ResultBuffers {
  ResultBuffers(std::size_t rows, std::size_t k, bool negate = false)
      : k(k),
        negate(negate),
        ids(rows * k, -1),
        distances(rows * k, negate ? -std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::infinity()) {}

  void put(std::size_t row, const std::vector<strata::Neighbor>& neighbors) {
    for (std::size_t i = 0; i < neighbors.size() && i < k; ++i) {
      ids[row * k + i] = neighbors[i].id;
      distances[row * k + i] = negate ? -neighbors[i].distance : neighbors[i].distance;
    }
  }

  SearchResult to_python(std::size_t rows, bool single) && {
    std::vector<std::size_t> shape =
        single ? std::vector<std::size_t>{k} : std::vector<std::size_t>{rows, k};
    return {to_numpy(std::move(ids), shape), to_numpy(std::move(distances), shape)};
  }

  std::size_t k;
  bool negate;
  std::vector<std::int64_t> ids;
  std::vector<float> distances;
};

// Runs search_one(row) for every row (in parallel when threads != 1) and collects padded results.
// Called with the GIL released.
template <typename SearchOne>
ResultBuffers search_rows(std::size_t rows, std::size_t k, std::optional<std::size_t> threads,
                          const SearchOne& search_one, bool negate = false) {
  if (threads && *threads == 0) {
    throw nb::value_error("threads must be positive (or None for all cores)");
  }
  ResultBuffers out(rows, k, negate);
  std::vector<std::optional<strata::Error>> errors(rows);
  auto run = [&](std::size_t row) {
    auto result = search_one(row);
    if (result) {
      out.put(row, *result);
    } else {
      errors[row] = result.error();
    }
  };
  const std::size_t n_threads = threads.value_or(0);
  if (n_threads == 1 || rows <= 1) {
    for (std::size_t row = 0; row < rows; ++row) {
      run(row);
    }
  } else {
    strata::ThreadPool pool(n_threads);
    pool.parallel_for(rows, run);
  }
  for (const auto& error : errors) {
    if (error) {
      raise(*error);
    }
  }
  return out;
}

nb::ndarray<nb::numpy, std::int64_t> id_range(std::size_t first, std::size_t count) {
  std::vector<std::int64_t> ids(count);
  for (std::size_t i = 0; i < count; ++i) {
    ids[i] = static_cast<std::int64_t>(first + i);
  }
  return to_numpy(std::move(ids), {count});
}

// --- Wrapped objects ----------

struct PyBruteForce {
  strata::BruteForceIndex index;
  mutable std::shared_mutex mutex;
};

struct PyAttributeTable {
  strata::AttributeTable table;
  std::uint64_t generation = 0;  // bumped on append; compiled filters check it
};

struct PyCompiledFilter {
  strata::CompiledFilter filter;
  const PyAttributeTable* table;
  std::uint64_t generation;

  void check_current() const {
    if (table->generation != generation) {
      throw std::runtime_error(
          "the AttributeTable changed after this filter was compiled; compile it again");
    }
  }
};

struct PyPqIndex {
  strata::PqIndex index;
  mutable std::shared_mutex mutex;
};

struct PyBm25 {
  strata::Bm25Index index;
  mutable std::shared_mutex mutex;
};

strata::AnalyzerConfig make_analyzer_config(const std::string& tokenizer, bool possessive,
                                            bool lowercase,
                                            const std::optional<std::string>& stopwords,
                                            const std::optional<std::string>& stemmer,
                                            std::size_t max_token_length) {
  strata::AnalyzerConfig config;
  if (tokenizer == "unicode") {
    config.tokenizer = strata::Tokenizer::kUnicodeWords;
  } else if (tokenizer == "plain") {
    config.tokenizer = strata::Tokenizer::kPlain;
  } else {
    throw nb::value_error("tokenizer must be 'unicode' or 'plain'");
  }
  config.english_possessive = possessive;
  config.lowercase = lowercase;
  if (!stopwords) {
    config.stopwords = strata::Stopwords::kNone;
  } else if (*stopwords == "lucene_english") {
    config.stopwords = strata::Stopwords::kLuceneEnglish;
  } else {
    throw nb::value_error("stopwords must be None or 'lucene_english'");
  }
  if (!stemmer) {
    config.stemmer = strata::Stemmer::kNone;
  } else if (*stemmer == "porter") {
    config.stemmer = strata::Stemmer::kPorter;
  } else {
    throw nb::value_error("stemmer must be None or 'porter'");
  }
  if (max_token_length == 0) {
    throw nb::value_error("max_token_length must be positive");
  }
  config.max_token_length = max_token_length;
  return config;
}

std::string describe(const strata::AnalyzerConfig& c) {
  return std::string("Analyzer(tokenizer='") +
         (c.tokenizer == strata::Tokenizer::kPlain ? "plain" : "unicode") +
         "', english_possessive=" + (c.english_possessive ? "True" : "False") +
         ", lowercase=" + (c.lowercase ? "True" : "False") + ", stopwords=" +
         (c.stopwords == strata::Stopwords::kLuceneEnglish ? "'lucene_english'" : "None") +
         ", stemmer=" + (c.stemmer == strata::Stemmer::kPorter ? "'porter'" : "None") +
         ", max_token_length=" + std::to_string(c.max_token_length) + ")";
}

#ifdef STRATA_HAS_HNSW
strata::NeighborSelection parse_selection(const std::string& name) {
  if (name == "heuristic") {
    return strata::NeighborSelection::kHeuristic;
  }
  if (name == "simple") {
    return strata::NeighborSelection::kSimple;
  }
  throw nb::value_error(("unknown selection '" + name + "' (heuristic or simple)").c_str());
}

struct PyHnsw {
  strata::HnswIndex index;
  mutable std::shared_mutex mutex;
};
#endif

// Allowed-id bitsets for filtered search, one per accepted filter form. Built with the GIL held:
// AttributeTable.append also runs under the GIL, so the table cannot change mid-evaluation.
// `size` is the index size read now; search_filtered re-checks it under the index lock.
strata::Bitset bitset_from(const PyCompiledFilter& f, std::size_t /*size*/) {
  f.check_current();
  return f.filter.evaluate();
}

strata::Bitset bitset_from(const BoolMask& mask, std::size_t /*size*/) {
  strata::Bitset bits(mask.shape(0));
  const bool* m = mask.data();
  for (std::size_t i = 0; i < mask.shape(0); ++i) {
    if (m[i]) {
      bits.set(i);
    }
  }
  return bits;
}

strata::Bitset bitset_from(const IdArray& ids, std::size_t size) {
  strata::Bitset bits(size);
  const std::int64_t* p = ids.data();
  for (std::size_t i = 0; i < ids.shape(0); ++i) {
    if (p[i] < 0 || static_cast<std::size_t>(p[i]) >= size) {
      throw nb::index_error(("id " + std::to_string(p[i]) + " out of range").c_str());
    }
    bits.set(static_cast<std::size_t>(p[i]));
  }
  return bits;
}

// Pre-filtered brute-force search shared by the three search_filtered overloads.
template <typename FilterArg>
SearchResult brute_force_filtered(const PyBruteForce& self, const FloatArray& queries,
                                  std::size_t k, const FilterArg& filter,
                                  std::optional<std::size_t> threads);

strata::AttributeValue to_value(const std::variant<std::int64_t, std::string>& v) {
  return std::visit([](const auto& x) -> strata::AttributeValue { return x; }, v);
}

template <typename FilterArg>
SearchResult brute_force_filtered(const PyBruteForce& self, const FloatArray& queries,
                                  std::size_t k, const FilterArg& filter,
                                  std::optional<std::size_t> threads) {
  const Rows q = as_rows(queries, "queries");
  const std::size_t size_now = shared(self.mutex, [&] { return self.index.size(); });
  const strata::Bitset allowed = bitset_from(filter, size_now);  // GIL held
  std::optional<ResultBuffers> out;
  {
    nb::gil_scoped_release release;
    const std::shared_lock lock(self.mutex);
    if (allowed.size() != self.index.size()) {
      throw nb::value_error(("filter covers " + std::to_string(allowed.size()) +
                             " ids but the index has " + std::to_string(self.index.size()) +
                             " vectors")
                                .c_str());
    }
    out.emplace(search_rows(q.view.rows(), k, threads, [&](std::size_t row) {
      return self.index.search_filtered(q.view.row(row), k, allowed);
    }));
  }
  return std::move(*out).to_python(q.view.rows(), q.single);
}

using PyValue = std::variant<std::int64_t, std::string>;

const char* kThreadsDoc = "threads: worker threads for the batch (None = all cores, 1 = serial).";

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = R"doc(Strata vector search: C++ core bindings.

Arrays: float32 C-contiguous inputs are used without copying; other dtypes/layouts are converted
with one copy. Results are NumPy arrays handed over from C++ without copying.

Search results are (ids, distances): int64 and float32 arrays of shape (k,) for one query or
(num_queries, k) for a batch. Missing results (fewer than k matches) are id -1, distance +inf.
Distances are "lower is closer" for every metric: squared L2, negated inner product, 1 - cosine.

Concurrency: search, batch, and training calls release the GIL, so Python threads run them in
parallel. Each index has a reader/writer lock: searches share it; add() and remove() take it
exclusively, so inserts are serialized (and wait for in-flight searches).)doc";

  m.attr("has_hnsw") =
#ifdef STRATA_HAS_HNSW
      true;
#else
      false;
#endif

  m.def(
      "build_info",
      [] {
        nb::dict info;
        info["kernel"] = std::string(strata::best_kernel_name());
        info["compiler"] = strata::build::kCompiler;
        info["fp_flags"] = strata::build::kFpFlags;
        info["target"] = strata::build::kTarget;
        info["build_type"] = strata::build::kBuildType;
        info["has_hnsw"] = nb::bool_(
#ifdef STRATA_HAS_HNSW
            true
#else
            false
#endif
        );
        return info;
      },
      "How this build computes distances: kernel family, compiler, FP flags, target. Two builds "
      "give bit-identical distances when these match.");

  // --- Distances ----------

  auto pair_distance = [](strata::Metric metric) {
    return [metric](const FloatVector& a, const FloatVector& b, const std::string& kernel) {
      if (a.shape(0) != b.shape(0)) {
        throw nb::value_error("vectors have different lengths");
      }
      const auto fn = strata::distance_function(metric, parse_kernel(kernel));
      return fn({a.data(), a.shape(0)}, {b.data(), b.shape(0)});
    };
  };
  m.def("l2_squared", pair_distance(strata::Metric::kL2), "a"_a, "b"_a, "kernel"_a = "best",
        "Squared Euclidean distance. kernel: 'best' (SIMD for this build) or 'scalar'.");
  m.def("inner_product", pair_distance(strata::Metric::kInnerProduct), "a"_a, "b"_a,
        "kernel"_a = "best", "Negated dot product (lower = more similar).");
  m.def("cosine_distance", pair_distance(strata::Metric::kCosine), "a"_a, "b"_a,
        "kernel"_a = "best", "1 - cosine similarity; 1 when either vector is zero.");

  m.def(
      "distances",
      [](const FloatArray& queries, const FloatArray& base, const std::string& metric,
         const std::string& kernel, std::optional<std::size_t> threads) {
        const Rows q = as_rows(queries, "queries");
        const Rows b = as_rows(base, "base");
        if (q.view.cols() != b.view.cols()) {
          throw nb::value_error("queries and base have different dimensions");
        }
        const auto fn = strata::distance_function(parse_metric(metric), parse_kernel(kernel));
        if (threads && *threads == 0) {
          throw nb::value_error("threads must be positive (or None for all cores)");
        }
        std::vector<float> out(q.view.rows() * b.view.rows());
        {
          nb::gil_scoped_release release;
          const std::size_t nb_rows = b.view.rows();
          auto row = [&](std::size_t i) {
            for (std::size_t j = 0; j < nb_rows; ++j) {
              out[i * nb_rows + j] = fn(q.view.row(i), b.view.row(j));
            }
          };
          strata::ThreadPool pool(threads.value_or(0));
          pool.parallel_for(q.view.rows(), row);
        }
        return to_numpy(std::move(out), {q.view.rows(), b.view.rows()});
      },
      "queries"_a, "base"_a, "metric"_a = "l2", "kernel"_a = "best", "threads"_a = nb::none(),
      "All pairwise distances, shape (num_queries, num_base). Releases the GIL.");

  // --- Brute force ----------

  nb::class_<PyBruteForce>(m, "BruteForceIndex", R"doc(Exact k-NN by scanning every vector.

Thread safety: search methods may run concurrently from many Python threads (GIL released,
shared lock). add() and remove() take an exclusive lock: inserts are serialized.)doc")
      .def(
          "__init__",
          [](PyBruteForce* self, std::size_t dim, const std::string& metric,
             const std::string& kernel) {
            new (self) PyBruteForce{unwrap(strata::BruteForceIndex::create(
                                        dim, parse_metric(metric), parse_kernel(kernel))),
                                    {}};
          },
          "dim"_a, "metric"_a = "l2", "kernel"_a = "best")
      .def(
          "add",
          [](PyBruteForce& self, const FloatArray& vectors) {
            const Rows rows = as_rows(vectors, "vectors");
            std::size_t first = 0;
            {
              nb::gil_scoped_release release;
              const std::unique_lock lock(self.mutex);
              first = self.index.size();
              unwrap(self.index.add_batch(rows.view));
            }
            return id_range(first, rows.view.rows());
          },
          "vectors"_a,
          "Add one vector (1-D) or a batch (2-D). Returns the assigned ids (dense, in insertion "
          "order). Takes the exclusive lock: concurrent adds are serialized.")
      .def(
          "remove",
          [](PyBruteForce& self, std::int64_t id) {
            if (id < 0 || id > std::numeric_limits<strata::VectorId>::max()) {
              throw nb::key_error(("no vector with id " + std::to_string(id)).c_str());
            }
            nb::gil_scoped_release release;
            const std::unique_lock lock(self.mutex);
            unwrap(self.index.remove(static_cast<strata::VectorId>(id)));
          },
          "id"_a, "Delete a vector (tombstone; its id is not reused). KeyError if absent.")
      .def(
          "search",
          [](const PyBruteForce& self, const FloatArray& queries, std::size_t k,
             std::optional<std::size_t> threads) {
            const Rows q = as_rows(queries, "queries");
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(q.view.rows(), k, threads, [&](std::size_t row) {
                return self.index.search(q.view.row(row), k);
              }));
            }
            return std::move(*out).to_python(q.view.rows(), q.single);
          },
          "queries"_a, "k"_a, "threads"_a = 1,
          "Exact k nearest neighbors of one query (1-D) or a batch (2-D). Returns (ids, "
          "distances). "
          "threads: worker threads for a batch (None = all cores). Releases the GIL.")
      .def(
          "search_batch",
          [](const PyBruteForce& self, const FloatArray& queries, std::size_t k,
             std::optional<std::size_t> threads) {
            const Rows q = as_rows(queries, "queries");
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(q.view.rows(), k, threads, [&](std::size_t row) {
                return self.index.search(q.view.row(row), k);
              }));
            }
            return std::move(*out).to_python(q.view.rows(), false);
          },
          "queries"_a, "k"_a, "threads"_a = nb::none(),
          (std::string("Batch search, always 2-D results. ") + kThreadsDoc).c_str())
      .def(
          "search_filtered",
          [](const PyBruteForce& self, const FloatArray& queries, std::size_t k,
             const PyCompiledFilter& filter, std::optional<std::size_t> threads) {
            return brute_force_filtered(self, queries, k, filter, threads);
          },
          "queries"_a, "k"_a, "filter"_a, "threads"_a = 1,
          "Pre-filtered exact search: only ids the filter allows are scored. filter is a "
          "CompiledFilter, a bool mask of length len(index), or an int array of allowed ids.")
      .def(
          "search_filtered",
          [](const PyBruteForce& self, const FloatArray& queries, std::size_t k, const IdArray& ids,
             std::optional<std::size_t> threads) {
            return brute_force_filtered(self, queries, k, ids, threads);
          },
          "queries"_a, "k"_a, "filter"_a, "threads"_a = 1)
      .def(
          "search_filtered",
          [](const PyBruteForce& self, const FloatArray& queries, std::size_t k,
             const BoolMask& mask, std::optional<std::size_t> threads) {
            return brute_force_filtered(self, queries, k, mask, threads);
          },
          "queries"_a, "k"_a, "filter"_a, "threads"_a = 1)
      .def(
          "vector",
          [](const PyBruteForce& self, std::int64_t id) {
            auto copy = shared(self.mutex, [&] {
              if (id < 0 || static_cast<std::size_t>(id) >= self.index.size()) {
                throw nb::index_error("id out of range");
              }
              const auto v = self.index.vector(static_cast<strata::VectorId>(id));
              return std::vector<float>(v.begin(), v.end());
            });
            const std::size_t n = copy.size();
            return to_numpy(std::move(copy), {n});
          },
          "id"_a, "Copy of a stored vector (deleted vectors keep their data).")
      .def("is_deleted",
           [](const PyBruteForce& self, std::int64_t id) {
             return shared(self.mutex, [&] {
               if (id < 0 || static_cast<std::size_t>(id) >= self.index.size()) {
                 throw nb::index_error("id out of range");
               }
               return self.index.is_deleted(static_cast<strata::VectorId>(id));
             });
           })
      .def("__len__",
           [](const PyBruteForce& self) {
             return shared(self.mutex, [&] { return self.index.size(); });
           })
      .def_prop_ro("dim", [](const PyBruteForce& self) { return self.index.dim(); })
      .def_prop_ro("metric",
                   [](const PyBruteForce& self) {
                     return std::string(strata::to_string(self.index.metric()));
                   })
      .def_prop_ro("live_size", [](const PyBruteForce& self) {
        return shared(self.mutex, [&] { return self.index.live_size(); });
      });

  // --- Attributes and filters ----------

  nb::class_<PyAttributeTable>(m, "AttributeTable", R"doc(Columnar metadata, one row per vector id.

schema: list of (name, type) with type "int" or "category". Row i describes vector id i.
Appending invalidates previously compiled filters (they raise and must be recompiled).
Not thread-safe for concurrent appends.)doc")
      .def(
          "__init__",
          [](PyAttributeTable* self,
             const std::vector<std::pair<std::string, std::string>>& schema) {
            std::vector<strata::ColumnSpec> specs;
            for (const auto& [name, type] : schema) {
              if (type != "int" && type != "category") {
                throw nb::value_error(
                    ("column type must be 'int' or 'category', got '" + type + "'").c_str());
              }
              specs.push_back(
                  {name, type == "int" ? strata::ColumnType::kInt : strata::ColumnType::kCategory});
            }
            new (self)
                PyAttributeTable{unwrap(strata::AttributeTable::create(std::move(specs))), 0};
          },
          "schema"_a)
      .def(
          "append",
          [](PyAttributeTable& self, const std::vector<PyValue>& row) {
            std::vector<strata::AttributeValue> values;
            for (const auto& v : row) {
              values.push_back(to_value(v));
            }
            unwrap(self.table.append(values));
            ++self.generation;
          },
          "row"_a, "Append the row for the next id; values in schema order.")
      .def(
          "extend",
          [](PyAttributeTable& self, const std::vector<std::vector<PyValue>>& rows) {
            for (const auto& row : rows) {
              std::vector<strata::AttributeValue> values;
              for (const auto& v : row) {
                values.push_back(to_value(v));
              }
              unwrap(self.table.append(values));
            }
            ++self.generation;
          },
          "rows"_a, "Append many rows.")
      .def("__len__", [](const PyAttributeTable& self) { return self.table.rows(); });

  nb::class_<PyCompiledFilter>(m, "CompiledFilter",
                               "A Filter resolved against one AttributeTable. Immutable; safe to "
                               "share between threads. Raises if the table was appended to since.")
      .def("matches",
           [](const PyCompiledFilter& self, std::int64_t id) {
             self.check_current();
             return id >= 0 && self.filter.matches(static_cast<strata::VectorId>(id));
           })
      .def(
          "evaluate",
          [](const PyCompiledFilter& self) {
            self.check_current();
            const strata::Bitset bits = self.filter.evaluate();
            std::vector<std::uint8_t> mask(bits.size(), 0);
            bits.for_each_set([&](std::size_t i) { mask[i] = 1; });
            return to_numpy_bool(std::move(mask));
          },
          "Boolean mask of matching ids.")
      .def(
          "estimate_selectivity",
          [](const PyCompiledFilter& self, std::size_t samples, std::uint64_t seed) {
            self.check_current();
            return self.filter.estimate_selectivity(samples, seed);
          },
          "samples"_a = 1024, "seed"_a = 1,
          "Fraction of ids that match, estimated from random samples (exact if samples >= rows).");

  nb::class_<strata::Filter>(m, "Filter", R"doc(Filter expression over AttributeTable columns.

Build with Filter.equals / range / in_ and combine with & (all), | (any), ~ (not), or
Filter.all_of / any_of / negate. compile(table) resolves it for fast evaluation.)doc")
      .def_static(
          "equals",
          [](const std::string& column, const PyValue& value) {
            return strata::Filter::equals(column, to_value(value));
          },
          "column"_a, "value"_a)
      .def_static("range", &strata::Filter::range, "column"_a, "lo"_a, "hi"_a,
                  "Inclusive integer range.")
      .def_static(
          "in_",
          [](const std::string& column, const std::vector<PyValue>& values) {
            std::vector<strata::AttributeValue> converted;
            for (const auto& v : values) {
              converted.push_back(to_value(v));
            }
            return strata::Filter::in(column, std::move(converted));
          },
          "column"_a, "values"_a)
      .def_static("all_of", &strata::Filter::all_of, "filters"_a)
      .def_static("any_of", &strata::Filter::any_of, "filters"_a)
      .def_static("negate", &strata::Filter::negate, "filter"_a)
      .def("__and__", [](const strata::Filter& a,
                         const strata::Filter& b) { return strata::Filter::all_of({a, b}); })
      .def("__or__", [](const strata::Filter& a,
                        const strata::Filter& b) { return strata::Filter::any_of({a, b}); })
      .def("__invert__", [](const strata::Filter& a) { return strata::Filter::negate(a); })
      .def(
          "compile",
          [](const strata::Filter& self, const PyAttributeTable& table) {
            return PyCompiledFilter{unwrap(strata::CompiledFilter::compile(self, table.table)),
                                    &table, table.generation};
          },
          "table"_a, nb::keep_alive<0, 2>(),
          "Resolve column names and categories against `table`. The result keeps the table alive.");

  // --- Product quantization ----------

  nb::class_<strata::ProductQuantizer>(m, "ProductQuantizer",
                                       R"doc(Product quantizer with 8-bit codes.

Each vector is split into m sub-vectors (dim must be divisible by m), each stored as one byte.
Immutable after training; safe to share between threads.)doc")
      .def_static(
          "train",
          [](const FloatArray& data, const std::string& metric, std::size_t m,
             std::size_t max_training_rows, std::size_t kmeans_iterations, std::uint64_t seed,
             std::optional<std::size_t> threads) {
            const Rows rows = as_rows(data, "data");
            const auto parsed = parse_metric(metric);
            nb::gil_scoped_release release;
            strata::ThreadPool pool(threads.value_or(0));
            return unwrap(strata::ProductQuantizer::train(rows.view, parsed,
                                                          {.m = m,
                                                           .max_training_rows = max_training_rows,
                                                           .kmeans_iterations = kmeans_iterations,
                                                           .seed = seed},
                                                          &pool));
          },
          "data"_a, "metric"_a = "l2", "m"_a = 16, "max_training_rows"_a = 65536,
          "kmeans_iterations"_a = 25, "seed"_a = 42, "threads"_a = nb::none(),
          "Train codebooks (k-means per subspace). Deterministic for a given seed. Releases the "
          "GIL.")
      .def(
          "encode",
          [](const strata::ProductQuantizer& self, const FloatArray& vectors) {
            const Rows rows = as_rows(vectors, "vectors");
            if (rows.view.cols() != self.dim()) {
              throw nb::value_error("wrong dimension");
            }
            const std::size_t m = self.code_size();
            std::vector<std::uint8_t> codes(rows.view.rows() * m);
            {
              nb::gil_scoped_release release;
              for (std::size_t i = 0; i < rows.view.rows(); ++i) {
                self.encode(rows.view.row(i), std::span(codes).subspan(i * m, m));
              }
            }
            return to_numpy(std::move(codes), rows.single
                                                  ? std::vector<std::size_t>{m}
                                                  : std::vector<std::size_t>{rows.view.rows(), m});
          },
          "vectors"_a, "uint8 codes, shape (m,) or (n, m).")
      .def(
          "decode",
          [](const strata::ProductQuantizer& self, const U8Array& codes) {
            const std::size_t m = self.code_size();
            const bool single = codes.ndim() == 1;
            if ((codes.ndim() != 1 && codes.ndim() != 2) || codes.shape(codes.ndim() - 1) != m) {
              throw nb::value_error("codes must have shape (m,) or (n, m)");
            }
            const std::size_t n = single ? 1 : codes.shape(0);
            std::vector<float> out(n * self.dim());
            for (std::size_t i = 0; i < n; ++i) {
              self.decode({codes.data() + i * m, m},
                          std::span(out).subspan(i * self.dim(), self.dim()));
            }
            return to_numpy(std::move(out), single ? std::vector<std::size_t>{self.dim()}
                                                   : std::vector<std::size_t>{n, self.dim()});
          },
          "codes"_a, "Reconstructions (for cosine: of the normalized vectors).")
      .def(
          "compute_table",
          [](const strata::ProductQuantizer& self, const FloatVector& query) {
            if (query.shape(0) != self.dim()) {
              throw nb::value_error("wrong dimension");
            }
            std::vector<float> table(self.m() * strata::ProductQuantizer::kCentroids);
            self.compute_table({query.data(), query.shape(0)}, table);
            return to_numpy(std::move(table), {self.m(), strata::ProductQuantizer::kCentroids});
          },
          "query"_a, "ADC lookup table, shape (m, 256).")
      .def_prop_ro(
          "codebooks",
          [](const strata::ProductQuantizer& self) {
            const auto data = self.codebooks().data();
            return to_numpy(std::vector<float>(data.begin(), data.end()),
                            {self.m(), strata::ProductQuantizer::kCentroids, self.sub_dim()});
          },
          // The array owns its (copied) buffer; don't tie it to the quantizer.
          nb::rv_policy::move, "Copy of the codebooks, shape (m, 256, sub_dim).")
      .def_prop_ro("dim", &strata::ProductQuantizer::dim)
      .def_prop_ro("m", &strata::ProductQuantizer::m)
      .def_prop_ro("sub_dim", &strata::ProductQuantizer::sub_dim)
      .def_prop_ro("code_size", &strata::ProductQuantizer::code_size)
      .def_prop_ro("metric", [](const strata::ProductQuantizer& self) {
        return std::string(strata::to_string(self.metric()));
      });

  nb::class_<PyPqIndex>(m, "PqIndex",
                        R"doc(Flat PQ index: ADC scan over codes, optional exact rerank.

keep_originals stores full-precision vectors (needed for rerank > 0).
Thread safety: searches may run concurrently (GIL released, shared lock); add() takes an
exclusive lock, so inserts are serialized.)doc")
      .def(
          "__init__",
          [](PyPqIndex* self, const strata::ProductQuantizer& quantizer, bool keep_originals) {
            new (self) PyPqIndex{unwrap(strata::PqIndex::create(quantizer, keep_originals)), {}};
          },
          "quantizer"_a, "keep_originals"_a = true)
      .def(
          "add",
          [](PyPqIndex& self, const FloatArray& vectors, std::optional<std::size_t> threads) {
            const Rows rows = as_rows(vectors, "vectors");
            std::size_t first = 0;
            {
              nb::gil_scoped_release release;
              const std::unique_lock lock(self.mutex);
              first = self.index.size();
              strata::ThreadPool pool(threads.value_or(0));
              unwrap(self.index.add_batch(rows.view, &pool));
            }
            return id_range(first, rows.view.rows());
          },
          "vectors"_a, "threads"_a = nb::none(),
          "Encode and add vectors; returns their ids. Exclusive lock: adds are serialized.")
      .def(
          "search",
          [](const PyPqIndex& self, const FloatArray& queries, std::size_t k, std::size_t rerank,
             std::optional<std::size_t> threads) {
            const Rows q = as_rows(queries, "queries");
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(q.view.rows(), k, threads, [&](std::size_t row) {
                return self.index.search(q.view.row(row), k, {.rerank = rerank});
              }));
            }
            return std::move(*out).to_python(q.view.rows(), q.single);
          },
          "queries"_a, "k"_a, "rerank"_a = 0, "threads"_a = 1,
          "rerank=0: ADC distances (estimates). rerank>=k: rerank the top `rerank` by ADC with "
          "exact distances. Returns (ids, distances). Releases the GIL.")
      .def("__len__",
           [](const PyPqIndex& self) {
             return shared(self.mutex, [&] { return self.index.size(); });
           })
      .def_prop_ro("code_bytes",
                   [](const PyPqIndex& self) {
                     return shared(self.mutex, [&] { return self.index.code_bytes(); });
                   })
      .def_prop_ro("original_bytes",
                   [](const PyPqIndex& self) {
                     return shared(self.mutex, [&] { return self.index.original_bytes(); });
                   })
      .def_prop_ro("codebook_bytes",
                   [](const PyPqIndex& self) { return self.index.codebook_bytes(); });

  // --- Text analysis and BM25 ----------

  nb::class_<strata::Analyzer>(
      m, "Analyzer", R"doc(Text analysis: tokenize, possessive filter, lowercase, stopwords, stem.

tokenizer: "unicode" (UAX #29 word boundaries, as Lucene's StandardTokenizer) or "plain"
(maximal runs of Unicode letters/digits). stopwords: None or "lucene_english" (Lucene's 33-word
list). stemmer: None or "porter" (Lucene's Porter stemmer). Analyzer.anserini_english() is
Anserini's default BEIR configuration; Analyzer.plain() is the plain tokenizer + lowercase.
Immutable; safe to share between threads.)doc")
      .def(
          "__init__",
          [](strata::Analyzer* self, const std::string& tokenizer, bool english_possessive,
             bool lowercase, const std::optional<std::string>& stopwords,
             const std::optional<std::string>& stemmer, std::size_t max_token_length) {
            new (self) strata::Analyzer(make_analyzer_config(
                tokenizer, english_possessive, lowercase, stopwords, stemmer, max_token_length));
          },
          "tokenizer"_a = "unicode", "english_possessive"_a = false, "lowercase"_a = true,
          "stopwords"_a = nb::none(), "stemmer"_a = nb::none(), "max_token_length"_a = 255)
      .def_static("anserini_english",
                  [] { return strata::Analyzer(strata::AnalyzerConfig::anserini_english()); })
      .def_static("plain", [] { return strata::Analyzer(strata::AnalyzerConfig::plain()); })
      .def(
          "analyze",
          [](const strata::Analyzer& self, const std::string& text) {
            nb::gil_scoped_release release;
            return self.analyze(text);
          },
          "text"_a, "Tokens of `text` (UTF-8), after every configured filter.")
      .def("__repr__", [](const strata::Analyzer& self) { return describe(self.config()); });

  m.def(
      "porter_stem", [](const std::string& word) { return strata::porter_stem(word); }, "word"_a,
      "Porter stem of one lowercase word (Lucene PorterStemmer semantics).");

  nb::class_<PyBm25>(m, "Bm25Index",
                     R"doc(BM25 over an inverted index, scored exactly as Lucene/Anserini.

idf = ln(1 + (N - df + 0.5) / (df + 0.5)); defaults k1=0.9, b=0.4 (Anserini BEIR). length_encoding
"lucene" uses Lucene's lossy 1-byte document lengths (needed to reproduce Anserini); "exact" uses
exact token counts (as bm25s does). The default analyzer is Analyzer.anserini_english().

Document ids are dense, in insertion order: the same id space as the vector indexes, so adding
document i to each index in the same order lines ids up for hybrid retrieval. remove() tombstones
an id without reusing it. Deleted documents still count in N, df, and avgdl (as in Lucene).

Search returns (ids, scores): higher is better; missing results are id -1, score -inf.
Thread safety: searches may run concurrently (GIL released, shared lock); add() and remove()
take an exclusive lock, so inserts are serialized.)doc")
      .def(
          "__init__",
          [](PyBm25* self, float k1, float b, const std::string& length_encoding,
             const std::optional<strata::Analyzer>& analyzer) {
            strata::Bm25Params params{.k1 = k1, .b = b};
            if (length_encoding == "lucene") {
              params.length_encoding = strata::LengthEncoding::kLucene;
            } else if (length_encoding == "exact") {
              params.length_encoding = strata::LengthEncoding::kExact;
            } else {
              throw nb::value_error("length_encoding must be 'lucene' or 'exact'");
            }
            if (analyzer) {
              params.analyzer = analyzer->config();
            }
            new (self) PyBm25{unwrap(strata::Bm25Index::create(params)), {}};
          },
          "k1"_a = 0.9F, "b"_a = 0.4F, "length_encoding"_a = "lucene", "analyzer"_a = nb::none())
      .def(
          "add",
          [](PyBm25& self, const std::vector<std::string>& texts) {
            std::size_t first = 0;
            {
              nb::gil_scoped_release release;
              const std::unique_lock lock(self.mutex);
              first = self.index.size();
              for (const auto& text : texts) {
                unwrap(self.index.add(text));
              }
            }
            return id_range(first, texts.size());
          },
          "texts"_a, "Analyze and index documents; returns their ids. Exclusive lock.")
      .def(
          "add",
          [](PyBm25& self, const std::string& text) {
            nb::gil_scoped_release release;
            const std::unique_lock lock(self.mutex);
            return static_cast<std::int64_t>(unwrap(self.index.add(text)));
          },
          "text"_a)
      .def(
          "add_tokens",
          [](PyBm25& self, const std::vector<std::vector<std::string>>& docs) {
            std::size_t first = 0;
            {
              nb::gil_scoped_release release;
              const std::unique_lock lock(self.mutex);
              first = self.index.size();
              for (const auto& tokens : docs) {
                unwrap(self.index.add_tokens(tokens));
              }
            }
            return id_range(first, docs.size());
          },
          "docs"_a, "Index pre-analyzed documents (lists of tokens), bypassing the analyzer.")
      .def(
          "remove",
          [](PyBm25& self, std::int64_t id) {
            if (id < 0 || id > std::numeric_limits<strata::VectorId>::max()) {
              throw nb::key_error(("no document with id " + std::to_string(id)).c_str());
            }
            nb::gil_scoped_release release;
            const std::unique_lock lock(self.mutex);
            unwrap(self.index.remove(static_cast<strata::VectorId>(id)));
          },
          "id"_a)
      .def(
          "search",
          [](const PyBm25& self, const std::string& query, std::size_t k) {
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(
                  1, k, 1, [&](std::size_t) { return self.index.search(query, k); }, true));
            }
            return std::move(*out).to_python(1, true);
          },
          "query"_a, "k"_a, "Top k documents for one query string: (ids, scores), shape (k,).")
      .def(
          "search",
          [](const PyBm25& self, const std::vector<std::string>& queries, std::size_t k,
             std::optional<std::size_t> threads) {
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(
                  queries.size(), k, threads,
                  [&](std::size_t row) { return self.index.search(queries[row], k); }, true));
            }
            return std::move(*out).to_python(queries.size(), false);
          },
          "queries"_a, "k"_a, "threads"_a = nb::none(),
          (std::string("Batch search over query strings: (ids, scores), shape (num_queries, k). ") +
           kThreadsDoc)
              .c_str())
      .def(
          "search_tokens",
          [](const PyBm25& self, const std::vector<std::string>& tokens, std::size_t k) {
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(
                  1, k, 1, [&](std::size_t) { return self.index.search_tokens(tokens, k); }, true));
            }
            return std::move(*out).to_python(1, true);
          },
          "tokens"_a, "k"_a, "Search with pre-analyzed query tokens (bypasses the analyzer).")
      .def(
          "search_tokens",
          [](const PyBm25& self, const std::vector<std::vector<std::string>>& queries,
             std::size_t k, std::optional<std::size_t> threads) {
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(
                  queries.size(), k, threads,
                  [&](std::size_t row) { return self.index.search_tokens(queries[row], k); },
                  true));
            }
            return std::move(*out).to_python(queries.size(), false);
          },
          "queries"_a, "k"_a, "threads"_a = nb::none())
      .def("__len__",
           [](const PyBm25& self) { return shared(self.mutex, [&] { return self.index.size(); }); })
      .def_prop_ro("live_size",
                   [](const PyBm25& self) {
                     return shared(self.mutex, [&] { return self.index.live_size(); });
                   })
      .def_prop_ro("vocabulary_size",
                   [](const PyBm25& self) {
                     return shared(self.mutex, [&] { return self.index.vocabulary_size(); });
                   })
      .def_prop_ro("total_terms",
                   [](const PyBm25& self) {
                     return shared(self.mutex, [&] { return self.index.total_terms(); });
                   })
      .def_prop_ro("doc_count",
                   [](const PyBm25& self) {
                     return shared(self.mutex, [&] { return self.index.doc_count(); });
                   })
      .def("is_deleted",
           [](const PyBm25& self, std::int64_t id) {
             return shared(self.mutex, [&] {
               return id >= 0 && self.index.is_deleted(static_cast<strata::VectorId>(id));
             });
           })
      .def("doc_length",
           [](const PyBm25& self, std::int64_t id) {
             return shared(self.mutex, [&] {
               if (id < 0 || static_cast<std::size_t>(id) >= self.index.size()) {
                 throw nb::index_error("id out of range");
               }
               return self.index.doc_length(static_cast<strata::VectorId>(id));
             });
           })
      .def("doc_freq",
           [](const PyBm25& self, const std::string& term) {
             return shared(self.mutex, [&] { return self.index.doc_freq(term); });
           })
      .def_prop_ro("analyzer", [](const PyBm25& self) { return self.index.analyzer(); })
      .def_prop_ro("k1", [](const PyBm25& self) { return self.index.params().k1; })
      .def_prop_ro("b", [](const PyBm25& self) { return self.index.params().b; })
      .def_prop_ro("length_encoding", [](const PyBm25& self) {
        return std::string(self.index.params().length_encoding == strata::LengthEncoding::kLucene
                               ? "lucene"
                               : "exact");
      });

  // --- Fusion ----------

  // Converts row `row` of each (ids, scores) pair into Neighbor lists (distance = -score),
  // skipping -1 padding. Scores must be higher-is-better similarities.
  using IdMatrix = nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  using ScoreMatrix = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

  auto check_shapes = [](const std::vector<IdMatrix>& ids, const std::vector<ScoreMatrix>* scores) {
    if (ids.empty()) {
      throw nb::value_error("need at least one result list");
    }
    const std::size_t rows = ids[0].shape(0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (ids[i].shape(0) != rows) {
        throw nb::value_error("every list must have the same number of queries (rows)");
      }
      if (scores != nullptr &&
          ((*scores)[i].shape(0) != rows || (*scores)[i].shape(1) != ids[i].shape(1))) {
        throw nb::value_error("scores must have the same shape as ids");
      }
    }
    return rows;
  };

  m.def(
      "fuse_rrf",
      [check_shapes](const std::vector<IdMatrix>& ids, std::size_t k, double rrf_k,
                     std::optional<std::size_t> threads) {
        const std::size_t rows = check_shapes(ids, nullptr);
        std::optional<ResultBuffers> out;
        {
          nb::gil_scoped_release release;
          out.emplace(search_rows(
              rows, k, threads,
              [&](std::size_t row) {
                std::vector<std::vector<strata::Neighbor>> lists(ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i) {
                  const std::size_t n = ids[i].shape(1);
                  const std::int64_t* r = ids[i].data() + row * n;
                  for (std::size_t j = 0; j < n && r[j] >= 0; ++j) {
                    lists[i].push_back(
                        {static_cast<strata::VectorId>(r[j]), static_cast<float>(j)});
                  }
                }
                return strata::reciprocal_rank_fusion(lists, k, rrf_k);
              },
              true));
        }
        return std::move(*out).to_python(rows, false);
      },
      "ids"_a, "k"_a, "rrf_k"_a = 60.0, "threads"_a = nb::none(),
      R"doc(Reciprocal rank fusion of ranked id lists.

ids: list of (num_queries, n_i) int64 arrays, each row ranked best first, -1 padding ignored
(e.g. the ids from a vector search and a BM25 search). score = sum 1 / (rrf_k + rank), rank from 1.
Returns (ids, scores) of shape (num_queries, k); ties by ascending id; padding -1 / -inf.)doc");

  m.def(
      "fuse_weighted",
      [check_shapes](const std::vector<IdMatrix>& ids, const std::vector<ScoreMatrix>& scores,
                     const std::vector<double>& weights, std::size_t k,
                     std::optional<std::size_t> threads) {
        if (scores.size() != ids.size()) {
          throw nb::value_error("one scores array per ids array required");
        }
        const std::size_t rows = check_shapes(ids, &scores);
        std::optional<ResultBuffers> out;
        {
          nb::gil_scoped_release release;
          out.emplace(search_rows(
              rows, k, threads,
              [&](std::size_t row) {
                std::vector<std::vector<strata::Neighbor>> lists(ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i) {
                  const std::size_t n = ids[i].shape(1);
                  const std::int64_t* r = ids[i].data() + row * n;
                  const float* s = scores[i].data() + row * n;
                  for (std::size_t j = 0; j < n && r[j] >= 0; ++j) {
                    lists[i].push_back({static_cast<strata::VectorId>(r[j]), -s[j]});
                  }
                }
                return strata::weighted_score_fusion(lists, weights, k);
              },
              true));
        }
        return std::move(*out).to_python(rows, false);
      },
      "ids"_a, "scores"_a, "weights"_a, "k"_a, "threads"_a = nb::none(),
      R"doc(Weighted fusion of min-max-normalized scores.

scores: higher-is-better similarities with the same shapes as ids (negate distances first).
Each list is min-max scaled to [0, 1] per query over its candidates; a document missing from a
list gets 0; fused = sum weights[i] * norm_i. Returns (ids, scores), shape (num_queries, k).)doc");

  // --- HNSW ----------
  // Bound only when src/index/hnsw.cpp exists; otherwise strata.HnswIndex (in __init__.py) raises
  // NotImplementedError.
#ifdef STRATA_HAS_HNSW
  nb::class_<PyHnsw>(m, "HnswIndex",
                     R"doc(Hierarchical Navigable Small World graph (approximate k-NN).

Thread safety: searches may run concurrently (GIL released, shared lock); add() takes an
exclusive lock, so inserts are serialized.)doc")
      .def(
          "__init__",
          [](PyHnsw* self, std::size_t dim, const std::string& metric, std::size_t M,
             std::size_t ef_construction, std::uint64_t seed, const std::string& selection) {
            new (self)
                PyHnsw{unwrap(strata::HnswIndex::create(dim, parse_metric(metric),
                                                        {.M = M,
                                                         .ef_construction = ef_construction,
                                                         .seed = seed,
                                                         .selection = parse_selection(selection)})),
                       {}};
          },
          "dim"_a, "metric"_a = "l2", "M"_a = 16, "ef_construction"_a = 200, "seed"_a = 42,
          "selection"_a = "heuristic",
          R"doc(selection: "heuristic" (the paper's Algorithm 4, default) or "simple" (closest M).)doc")
      .def(
          "add",
          [](PyHnsw& self, const FloatArray& vectors) {
            const Rows rows = as_rows(vectors, "vectors");
            std::size_t first = 0;
            {
              nb::gil_scoped_release release;
              const std::unique_lock lock(self.mutex);
              first = self.index.size();
              unwrap(self.index.add_batch(rows.view));
            }
            return id_range(first, rows.view.rows());
          },
          "vectors"_a, "Insert vectors; returns their ids. Exclusive lock: adds are serialized.")
      .def(
          "search",
          [](const PyHnsw& self, const FloatArray& queries, std::size_t k, std::size_t ef_search,
             std::optional<std::size_t> threads) {
            const Rows q = as_rows(queries, "queries");
            std::optional<ResultBuffers> out;
            {
              nb::gil_scoped_release release;
              const std::shared_lock lock(self.mutex);
              out.emplace(search_rows(q.view.rows(), k, threads, [&](std::size_t row) {
                return self.index.search(q.view.row(row), k, ef_search);
              }));
            }
            return std::move(*out).to_python(q.view.rows(), q.single);
          },
          "queries"_a, "k"_a, "ef_search"_a = 64, "threads"_a = 1)
      .def("__len__",
           [](const PyHnsw& self) { return shared(self.mutex, [&] { return self.index.size(); }); })
      .def_prop_ro("dim", [](const PyHnsw& self) { return self.index.dim(); })
      .def_prop_ro("M", [](const PyHnsw& self) { return self.index.params().M; })
      .def_prop_ro("ef_construction",
                   [](const PyHnsw& self) { return self.index.params().ef_construction; })
      .def_prop_ro("selection",
                   [](const PyHnsw& self) {
                     return self.index.params().selection == strata::NeighborSelection::kSimple
                                ? "simple"
                                : "heuristic";
                   })
      .def_prop_ro("max_level",
                   [](const PyHnsw& self) {
                     return shared(self.mutex, [&] { return self.index.max_level(); });
                   })
      .def_prop_ro("entry_point",
                   [](const PyHnsw& self) -> std::optional<std::int64_t> {
                     const auto ep = shared(self.mutex, [&] { return self.index.entry_point(); });
                     return ep ? std::optional<std::int64_t>(*ep) : std::nullopt;
                   })
      .def("level",
           [](const PyHnsw& self, std::int64_t id) {
             return shared(self.mutex, [&] {
               if (id < 0 || static_cast<std::size_t>(id) >= self.index.size()) {
                 throw nb::index_error("id out of range");
               }
               return self.index.level(static_cast<strata::VectorId>(id));
             });
           })
      .def("neighbors", [](const PyHnsw& self, std::int64_t id, int layer) {
        auto out = shared(self.mutex, [&] {
          if (id < 0 || static_cast<std::size_t>(id) >= self.index.size() || layer < 0 ||
              layer > self.index.level(static_cast<strata::VectorId>(id))) {
            throw nb::index_error("id or layer out of range");
          }
          const auto nbrs = self.index.neighbors(static_cast<strata::VectorId>(id), layer);
          return std::vector<std::int64_t>(nbrs.begin(), nbrs.end());
        });
        const std::size_t n = out.size();
        return to_numpy(std::move(out), {n});
      });
#endif
}
