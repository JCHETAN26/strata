// HNSW deletes (tombstones) and persistence (snapshot save/load). Compiled with hnsw_test.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <version>  // _LIBCPP_VERSION: which standard library this is

#include "hnsw_test_util.hpp"
#include "strata/brute_force.hpp"
#include "strata/crc32c.hpp"
#include "strata/hnsw.hpp"
#include "strata/recall.hpp"
#include "strata/snapshot.hpp"
#include "test_util.hpp"
#include "util/bytes.hpp"

namespace strata {
namespace {

namespace fs = std::filesystem;

HnswIndex make_index(std::size_t dim, HnswParams params = {}, Metric metric = Metric::kL2) {
  auto index = HnswIndex::create(dim, metric, params);
  EXPECT_TRUE(index.has_value()) << index.error().message;
  return std::move(*index);
}

void expect_same_graph(const HnswIndex& a, const HnswIndex& b) {
  EXPECT_TRUE(test::same_hnsw_graph(a, b));
}

void expect_same_results(const HnswIndex& a, const HnswIndex& b, const Matrix<float>& queries,
                         std::size_t k, std::size_t ef) {
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto ra = a.search(queries.row(q), k, ef);
    auto rb = b.search(queries.row(q), k, ef);
    ASSERT_TRUE(ra && rb);
    ASSERT_EQ(*ra, *rb) << "query " << q;  // ids and exact float distances
  }
}

class TempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("strata_hnsw_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
            "_" + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  fs::path dir_;
};

// The comparison used throughout must be able to fail: different seeds give different graphs.
TEST(HnswTestUtil, SameGraphDetectsDifferences) {
  const auto data = test::random_matrix(300, 8, 60);
  auto a = make_index(8, {.seed = 1});
  auto b = make_index(8, {.seed = 2});
  auto c = make_index(8, {.seed = 1});
  ASSERT_TRUE(a.add_batch(data) && b.add_batch(data) && c.add_batch(data));
  EXPECT_FALSE(test::same_hnsw_graph(a, b));
  EXPECT_TRUE(test::same_hnsw_graph(a, c));
  ASSERT_TRUE(c.remove(5));
  EXPECT_FALSE(test::same_hnsw_graph(a, c));  // tombstones count
}

// --- Deletes -------------------------------------------------------------------------------------

TEST(HnswDelete, RemoveErrorsAndCounts) {
  auto index = make_index(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(10, 4, 1)));
  EXPECT_EQ(index.remove(10).error().code, ErrorCode::kNotFound);
  ASSERT_TRUE(index.remove(3));
  EXPECT_EQ(index.remove(3).error().code, ErrorCode::kNotFound);
  EXPECT_TRUE(index.is_deleted(3));
  EXPECT_FALSE(index.is_deleted(4));
  EXPECT_EQ(index.size(), 10U);
  EXPECT_EQ(index.live_size(), 9U);
}

// Deletes the same ids from both indexes.
template <typename Pred>
void remove_where(HnswIndex& index, BruteForceIndex& exact, Pred pred) {
  for (VectorId id = 0; id < index.size(); ++id) {
    if (pred(id)) {
      ASSERT_TRUE(index.remove(id));
      ASSERT_TRUE(exact.remove(id));
    }
  }
}

// Fixture data for delete tests: 3000 random vectors, 30% deleted, plus an exact index over the
// live ones.
struct Deleted30 {
  Matrix<float> base = test::random_matrix(3000, 32, 30);
  Matrix<float> queries = test::random_matrix(100, 32, 31);
  HnswIndex index = make_index(32);
  BruteForceIndex exact = *BruteForceIndex::create(32, Metric::kL2);

  Deleted30() {
    EXPECT_TRUE(index.add_batch(base));
    EXPECT_TRUE(exact.add_batch(base));
    remove_where(index, exact, [](VectorId id) { return id % 10 < 3; });
  }
};

TEST(HnswDelete, DeletedNeverReturnedAndRecallOnLiveSet) {
  Deleted30 d;
  std::vector<std::vector<Neighbor>> results;
  std::vector<float> kth;
  for (std::size_t q = 0; q < d.queries.rows(); ++q) {
    auto found = d.index.search(d.queries.row(q), 10, 128);
    auto truth = d.exact.search(d.queries.row(q), 10);
    ASSERT_TRUE(found && truth);
    ASSERT_EQ(found->size(), 10U);
    for (const auto& n : *found) {
      ASSERT_FALSE(d.index.is_deleted(n.id)) << "deleted id " << n.id << " returned";
    }
    EXPECT_TRUE(std::ranges::is_sorted(*found));
    kth.push_back(truth->back().distance);
    results.push_back(std::move(*found));
  }
  EXPECT_GE(*recall_at_k_with_ties(results, kth, 10), 0.95);
}

TEST(HnswDelete, DeletedEntryPointStillNavigates) {
  auto index = make_index(16);
  ASSERT_TRUE(index.add_batch(test::random_matrix(1000, 16, 32)));
  const VectorId entry = *index.entry_point();
  ASSERT_TRUE(index.remove(entry));
  const auto queries = test::random_matrix(20, 16, 33);
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto found = index.search(queries.row(q), 10, 32);
    ASSERT_TRUE(found);
    ASSERT_EQ(found->size(), 10U);
    for (const auto& n : *found) {
      EXPECT_NE(n.id, entry);
    }
  }
}

// Under heavy deletion the beam keeps widening until it holds ef live nodes, so a search still
// returns k results, not fewer, even with ef_search == k.
TEST(HnswDelete, HeavyDeletionStillReturnsKLiveResults) {
  const auto base = test::random_matrix(2000, 16, 34);
  const auto queries = test::random_matrix(50, 16, 35);
  auto index = make_index(16);
  auto exact = *BruteForceIndex::create(16, Metric::kL2);
  ASSERT_TRUE(index.add_batch(base));
  ASSERT_TRUE(exact.add_batch(base));
  remove_where(index, exact, [](VectorId id) { return id % 20 != 0; });  // keep 5%: 100 live
  ASSERT_EQ(index.live_size(), 100U);
  std::vector<std::vector<Neighbor>> results;
  std::vector<float> kth;
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto found = index.search(queries.row(q), 10, 10);
    ASSERT_TRUE(found);
    ASSERT_EQ(found->size(), 10U) << "query " << q;
    for (const auto& n : *found) {
      ASSERT_EQ(n.id % 20, 0U);
    }
    kth.push_back(exact.search(queries.row(q), 10)->back().distance);
    results.push_back(std::move(*found));
  }
  EXPECT_GE(*recall_at_k_with_ties(results, kth, 10), 0.9);
}

// Fewer live nodes than k: every live node comes back, exactly as brute force would order them.
TEST(HnswDelete, FewerLiveThanKReturnsAllLive) {
  const auto base = test::random_matrix(500, 8, 36);
  auto index = make_index(8);
  auto exact = *BruteForceIndex::create(8, Metric::kL2);
  ASSERT_TRUE(index.add_batch(base));
  ASSERT_TRUE(exact.add_batch(base));
  remove_where(index, exact, [](VectorId id) { return id % 71 != 5; });  // 7 survivors
  ASSERT_EQ(index.live_size(), 7U);
  const auto query = test::random_matrix(1, 8, 37);
  auto found = index.search(query.row(0), 10, 10);
  ASSERT_TRUE(found);
  EXPECT_EQ(*found, *exact.search(query.row(0), 10));
}

TEST(HnswDelete, DeletingEverythingGivesEmptyResults) {
  auto index = make_index(4);
  ASSERT_TRUE(index.add_batch(test::random_matrix(50, 4, 38)));
  for (VectorId id = 0; id < 50; ++id) {
    ASSERT_TRUE(index.remove(id));
  }
  auto found = index.search(std::vector<float>{0, 0, 0, 0}, 5, 10);
  ASSERT_TRUE(found);
  EXPECT_TRUE(found->empty());
}

// Deletes only set tombstones; inserts still link to deleted nodes. So the graph does not depend
// on which deletes happened, only on what was inserted.
TEST(HnswDelete, DeletesDoNotChangeTheGraph) {
  const auto first = test::random_matrix(800, 16, 39);
  const auto second = test::random_matrix(800, 16, 40);
  auto with_deletes = make_index(16, {.ef_construction = 64});
  auto without = make_index(16, {.ef_construction = 64});
  ASSERT_TRUE(with_deletes.add_batch(first));
  ASSERT_TRUE(without.add_batch(first));
  for (VectorId id = 0; id < 800; id += 3) {
    ASSERT_TRUE(with_deletes.remove(id));
  }
  ASSERT_TRUE(with_deletes.add_batch(second));
  ASSERT_TRUE(without.add_batch(second));
  for (VectorId id = 0; id < 800; id += 3) {
    ASSERT_TRUE(without.remove(id));  // same tombstones, applied after the fact
  }
  expect_same_graph(with_deletes, without);
}

// --- Persistence ---------------------------------------------------------------------------------

struct PersistCase {
  Metric metric;
  NeighborSelection selection;
};

void PrintTo(const PersistCase& c, std::ostream* os) {
  *os << to_string(c.metric) << "_"
      << (c.selection == NeighborSelection::kSimple ? "simple" : "heuristic");
}

class HnswPersistence : public TempDir, public ::testing::WithParamInterface<PersistCase> {};

TEST_P(HnswPersistence, SaveThenLoadIsBitIdentical) {
  const auto [metric, selection] = GetParam();
  auto index = make_index(16, {.ef_construction = 100, .selection = selection}, metric);
  ASSERT_TRUE(index.add_batch(test::random_matrix(1500, 16, 41)));
  for (VectorId id = 0; id < 1500; id += 7) {
    ASSERT_TRUE(index.remove(id));
  }
  ASSERT_TRUE(index.save(dir_ / "hnsw.snap"));
  auto loaded = HnswIndex::load(dir_ / "hnsw.snap");
  ASSERT_TRUE(loaded) << loaded.error().message;
  expect_same_graph(index, *loaded);
  expect_same_results(index, *loaded, test::random_matrix(50, 16, 42), 10, 64);
}

// The core promise: saving is invisible to future inserts. Without the level generator's state
// in the snapshot, the second half would get different levels and this would fail.
TEST_P(HnswPersistence, SaveLoadThenAddEqualsNeverSaving) {
  const auto [metric, selection] = GetParam();
  const HnswParams params{.ef_construction = 100, .selection = selection};
  const auto first = test::random_matrix(1000, 16, 43);
  const auto second = test::random_matrix(1000, 16, 44);

  auto saved = make_index(16, params, metric);
  ASSERT_TRUE(saved.add_batch(first));
  ASSERT_TRUE(saved.remove(17));
  ASSERT_TRUE(saved.save(dir_ / "hnsw.snap"));
  auto resumed = HnswIndex::load(dir_ / "hnsw.snap");
  ASSERT_TRUE(resumed) << resumed.error().message;
  ASSERT_TRUE(resumed->add_batch(second));
  ASSERT_TRUE(resumed->remove(1500));

  auto straight = make_index(16, params, metric);
  ASSERT_TRUE(straight.add_batch(first));
  ASSERT_TRUE(straight.remove(17));
  ASSERT_TRUE(straight.add_batch(second));
  ASSERT_TRUE(straight.remove(1500));

  expect_same_graph(*resumed, straight);
  expect_same_results(*resumed, straight, test::random_matrix(50, 16, 45), 10, 64);
}

INSTANTIATE_TEST_SUITE_P(
    AllMetricsBothModes, HnswPersistence,
    ::testing::Values(PersistCase{Metric::kL2, NeighborSelection::kHeuristic},
                      PersistCase{Metric::kInnerProduct, NeighborSelection::kHeuristic},
                      PersistCase{Metric::kCosine, NeighborSelection::kHeuristic},
                      PersistCase{Metric::kL2, NeighborSelection::kSimple}),
    test::PrintedName{});

class HnswSnapshotFile : public TempDir {};

TEST_F(HnswSnapshotFile, EmptyIndexRoundTripsAndKeepsBuilding) {
  auto empty = make_index(8);
  ASSERT_TRUE(empty.save(dir_ / "hnsw.snap"));
  auto loaded = HnswIndex::load(dir_ / "hnsw.snap");
  ASSERT_TRUE(loaded) << loaded.error().message;
  EXPECT_EQ(loaded->size(), 0U);
  EXPECT_EQ(loaded->entry_point(), std::nullopt);
  EXPECT_EQ(loaded->max_level(), -1);
  const auto data = test::random_matrix(200, 8, 46);
  ASSERT_TRUE(loaded->add_batch(data));
  ASSERT_TRUE(empty.add_batch(data));
  expect_same_graph(*loaded, empty);
}

TEST_F(HnswSnapshotFile, FlatSnapshotIsNotAnHnswIndex) {
  Snapshot flat;
  flat.vectors = test::random_matrix(3, 4, 47);
  flat.deleted.assign(3, 0);
  ASSERT_TRUE(write_snapshot(dir_ / "flat.snap", flat));
  auto loaded = HnswIndex::load(dir_ / "flat.snap");
  ASSERT_FALSE(loaded);
  EXPECT_EQ(loaded.error().code, ErrorCode::kInvalidArgument);
}

// Byte-level edits to a saved file. The CRC is recomputed after each edit so the edit reaches the
// structural checks instead of being caught by the checksum.
class HnswSnapshotCorruption : public TempDir {
 protected:
  static constexpr std::size_t kCount = 300;
  static constexpr std::size_t kDim = 8;
  static constexpr std::size_t kHeader = 64;  // snapshot format versions 2 and 3

  void SetUp() override {
    TempDir::SetUp();
    auto index = make_index(kDim, {.M = 8, .ef_construction = 50});
    ASSERT_TRUE(index.add_batch(test::random_matrix(kCount, kDim, 48)));
    ASSERT_TRUE(index.save(path()));
    bytes_ = read_all();
  }

  [[nodiscard]] fs::path path() const { return dir_ / "hnsw.snap"; }
  [[nodiscard]] std::vector<std::byte> read_all() const {
    std::ifstream in(path(), std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
    std::vector<std::byte> out(raw.size());
    util::copy_bytes(out.data(), raw.data(), raw.size());
    return out;
  }
  // Writes `bytes` back with a fresh CRC.
  void write_fixed(std::vector<std::byte> bytes) const {
    const std::uint32_t crc = crc32c(std::span(bytes).first(bytes.size() - 4));
    util::copy_bytes(bytes.data() + bytes.size() - 4, &crc, 4);
    std::ofstream out(path(), std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  template <typename T>
  void patch(std::size_t offset, T value) {
    auto copy = bytes_;
    util::copy_bytes(copy.data() + offset, &value, sizeof(T));
    write_fixed(copy);
  }
  // Start of the HNSW index section, and of its layer-0 lists (after 48 bytes of fixed fields,
  // ending with the u64 generator draw count, then one level byte per node).
  [[nodiscard]] static std::size_t graph_start() {
    return kHeader + (kCount * kDim * 4) + ((kCount + 7) / 8);
  }
  [[nodiscard]] static std::size_t layer0_start() { return graph_start() + 48 + kCount; }
  void expect_corrupt(const std::string& fragment) const {
    auto loaded = HnswIndex::load(path());
    ASSERT_FALSE(loaded) << "loaded a damaged snapshot";
    EXPECT_EQ(loaded.error().code, ErrorCode::kCorruptData);
    EXPECT_NE(loaded.error().message.find(fragment), std::string::npos) << loaded.error().message;
  }

  std::vector<std::byte> bytes_;
};

TEST_F(HnswSnapshotCorruption, FlippedByteFailsTheChecksum) {
  for (std::size_t offset :
       {std::size_t{9}, graph_start() + 3, layer0_start() + 5, bytes_.size() - 10}) {
    auto copy = bytes_;
    copy[offset] ^= std::byte{0x40};
    std::ofstream out(path(), std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(copy.data()),
              static_cast<std::streamsize>(copy.size()));
    out.close();
    expect_corrupt("checksum");
  }
}

TEST_F(HnswSnapshotCorruption, UnsupportedSnapshotVersion) {
  patch<std::uint32_t>(8, 4);
  expect_corrupt("unsupported version");
}

TEST_F(HnswSnapshotCorruption, OppositeByteOrderIsRejected) {
  patch<std::uint32_t>(12, 0x04030201);
  expect_corrupt("opposite byte order");
}

TEST_F(HnswSnapshotCorruption, UnsupportedGraphVersion) {
  patch<std::uint32_t>(graph_start(), 3);
  expect_corrupt("unsupported graph version");
}

TEST_F(HnswSnapshotCorruption, EntryPointOutOfRange) {
  patch<std::uint32_t>(graph_start() + 36, kCount + 5);
  expect_corrupt("entry point");
}

TEST_F(HnswSnapshotCorruption, NeighborIdOutOfRange) {
  patch<std::uint32_t>(layer0_start() + 4, kCount + 100);  // node 0's first neighbor
  expect_corrupt("neighbor not on its layer");
}

TEST_F(HnswSnapshotCorruption, NeighborCountOverCapacity) {
  patch<std::uint32_t>(layer0_start(), 999);  // node 0's count; capacity is 2M = 16
  expect_corrupt("over capacity");
}

TEST_F(HnswSnapshotCorruption, TruncatedOrPaddedIndexSection) {
  auto snapshot = read_snapshot(path());
  ASSERT_TRUE(snapshot);
  auto truncated = *snapshot;
  truncated.index_data.resize(truncated.index_data.size() - 4);
  auto r1 = HnswIndex::from_snapshot(truncated);
  ASSERT_FALSE(r1);
  EXPECT_NE(r1.error().message.find("truncated"), std::string::npos) << r1.error().message;
  auto padded = *snapshot;
  padded.index_data.push_back(std::byte{0});
  auto r2 = HnswIndex::from_snapshot(padded);
  ASSERT_FALSE(r2);
  EXPECT_NE(r2.error().message.find("trailing"), std::string::npos) << r2.error().message;
}

// --- Cross-machine golden snapshots ----------------------------------------------------------
//
// tests/golden/hnsw_v3.snap (snapshot format version 3) was written on the Mac (M2, NEON) and is
// committed. Every machine that runs this suite (Linux/ARM, Linux/x86 with AVX2) must load it and
// reproduce the stored search results bit for bit, and rebuilding the same graph from its vectors
// must give the same graph. The vectors are small integers, so every SIMD kernel computes exact,
// identical distances. Regenerate (after a deliberate format change) with STRATA_WRITE_GOLDEN=1.
//
// tests/golden/hnsw_v2_libcxx.snap is the same index in format version 2, also written on the Mac.
// Version 2 stored the level generator as stream text, which libc++ and libstdc++ format
// differently, so it must load under libc++ (and upgrade to version 3 on save) and must be
// rejected with a clear error under libstdc++.

const fs::path kGoldenDir = fs::path(STRATA_TEST_SOURCE_DIR) / "golden";
constexpr std::size_t kGoldenCount = 400;
constexpr std::size_t kGoldenDim = 8;
constexpr std::size_t kGoldenK = 10;
constexpr std::size_t kGoldenEf = 32;
const HnswParams kGoldenParams{.M = 8, .ef_construction = 64, .seed = 7};

// Integer-valued vectors from the raw engine output (fixed by the standard, unlike the
// distributions), so they are the same on every standard library.
Matrix<float> golden_matrix(std::size_t rows, std::uint32_t seed) {
  std::mt19937 rng(seed);
  Matrix<float> m(rows, kGoldenDim);
  for (float& x : m.data()) {
    x = static_cast<float>(rng() % 16);
  }
  return m;
}

bool golden_deleted(VectorId id) { return id % 9 == 4; }

// The golden index, built from scratch on this machine.
HnswIndex golden_index() {
  auto index = make_index(kGoldenDim, kGoldenParams);
  EXPECT_TRUE(index.add_batch(golden_matrix(kGoldenCount, 1)));
  for (VectorId id = 0; id < kGoldenCount; ++id) {
    if (golden_deleted(id)) {
      EXPECT_TRUE(index.remove(id));
    }
  }
  return index;
}

// Expected results file: u32 nq, u32 k, then nq * k pairs of (u32 id, f32 distance), little-endian.
void write_expected_results(const HnswIndex& index, const fs::path& path) {
  const auto queries = golden_matrix(20, 2);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const auto nq = static_cast<std::uint32_t>(queries.rows());
  const auto k = static_cast<std::uint32_t>(kGoldenK);
  out.write(reinterpret_cast<const char*>(&nq), 4);
  out.write(reinterpret_cast<const char*>(&k), 4);
  for (std::size_t q = 0; q < queries.rows(); ++q) {
    auto found = index.search(queries.row(q), kGoldenK, kGoldenEf);
    ASSERT_TRUE(found);
    ASSERT_EQ(found->size(), kGoldenK);
    for (const auto& n : *found) {
      out.write(reinterpret_cast<const char*>(&n.id), 4);
      out.write(reinterpret_cast<const char*>(&n.distance), 4);
    }
  }
}

// Search results must match the stored ones: ids, and distances bit for bit.
void expect_expected_results(const HnswIndex& index, const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in) << "missing " << path;
  std::uint32_t nq = 0;
  std::uint32_t k = 0;
  in.read(reinterpret_cast<char*>(&nq), 4);
  in.read(reinterpret_cast<char*>(&k), 4);
  const auto queries = golden_matrix(nq, 2);
  for (std::size_t q = 0; q < nq; ++q) {
    auto found = index.search(queries.row(q), k, kGoldenEf);
    ASSERT_TRUE(found);
    ASSERT_EQ(found->size(), k);
    for (const auto& n : *found) {
      std::uint32_t id = 0;
      float distance = 0;
      in.read(reinterpret_cast<char*>(&id), 4);
      in.read(reinterpret_cast<char*>(&distance), 4);
      ASSERT_TRUE(in) << "expected results too short";
      EXPECT_EQ(n.id, id) << "query " << q;
      EXPECT_EQ(std::bit_cast<std::uint32_t>(n.distance), std::bit_cast<std::uint32_t>(distance))
          << "query " << q << ": " << n.distance << " vs " << distance;
    }
  }
}

TEST(HnswGolden, WriteFixture) {
  const char* flag = std::getenv("STRATA_WRITE_GOLDEN");
  if (flag == nullptr || std::string(flag) != "1") {
    GTEST_SKIP() << "set STRATA_WRITE_GOLDEN=1 to regenerate tests/golden/hnsw_v3*";
  }
  const auto index = golden_index();
  fs::create_directories(kGoldenDir);
  ASSERT_TRUE(index.save(kGoldenDir / "hnsw_v3.snap"));
  write_expected_results(index, kGoldenDir / "hnsw_v3_expected.bin");
}

TEST(HnswGolden, LoadsBitIdenticallyOnThisMachine) {
  auto loaded = HnswIndex::load(kGoldenDir / "hnsw_v3.snap");
  ASSERT_TRUE(loaded) << loaded.error().message;
  ASSERT_EQ(loaded->size(), kGoldenCount);
  expect_expected_results(*loaded, kGoldenDir / "hnsw_v3_expected.bin");
  // Building the same graph here, from the same vectors, seed, and order, must reproduce the file:
  // this checks that level assignment and neighbor selection are platform-independent.
  expect_same_graph(golden_index(), *loaded);
  // And the generator position must restore too: more inserts must keep the two identical.
  auto rebuilt = golden_index();
  const auto more = golden_matrix(100, 3);
  ASSERT_TRUE(loaded->add_batch(more));
  ASSERT_TRUE(rebuilt.add_batch(more));
  expect_same_graph(rebuilt, *loaded);
}

class HnswGoldenLegacy : public TempDir {};

TEST_F(HnswGoldenLegacy, Version2LoadsOnlyOnTheLibraryThatWroteIt) {
  auto loaded = HnswIndex::load(kGoldenDir / "hnsw_v2_libcxx.snap");
#if defined(_LIBCPP_VERSION)
  ASSERT_TRUE(loaded) << loaded.error().message;
  expect_expected_results(*loaded, kGoldenDir / "hnsw_v2_libcxx_expected.bin");
  expect_same_graph(golden_index(), *loaded);
  // Saving upgrades it to version 3, and the upgraded file keeps building exactly like a fresh
  // index, so the draw count recovered from the old text state is right.
  ASSERT_TRUE(loaded->save(dir_ / "upgraded.snap"));
  std::ifstream header(dir_ / "upgraded.snap", std::ios::binary);
  std::uint32_t version = 0;
  header.seekg(8);
  header.read(reinterpret_cast<char*>(&version), 4);
  EXPECT_EQ(version, 3U);
  auto upgraded = HnswIndex::load(dir_ / "upgraded.snap");
  ASSERT_TRUE(upgraded) << upgraded.error().message;
  auto rebuilt = golden_index();
  const auto more = golden_matrix(100, 3);
  ASSERT_TRUE(upgraded->add_batch(more));
  ASSERT_TRUE(rebuilt.add_batch(more));
  expect_same_graph(rebuilt, *upgraded);
#else
  ASSERT_FALSE(loaded) << "a libc++-written version 2 snapshot loaded under another library";
  EXPECT_EQ(loaded.error().code, ErrorCode::kCorruptData);
  EXPECT_NE(loaded.error().message.find("written by a different one"), std::string::npos)
      << loaded.error().message;
#endif
}

// The cross-library failure, reproduced on whichever library runs the test: rewrite the version 2
// generator text into the other library's shape (libc++ writes 312 numbers, libstdc++ 313) and
// check that it is rejected with the explanation, not misread.
TEST_F(HnswGoldenLegacy, OtherLibrarysGeneratorTextIsRejectedClearly) {
  auto snapshot = read_snapshot(kGoldenDir / "hnsw_v2_libcxx.snap");
  ASSERT_TRUE(snapshot) << snapshot.error().message;
  auto& section = snapshot->index_data;
  constexpr std::size_t kTextLength = 40;  // offset of the u32 text length in section version 1
  std::uint32_t length = 0;
  util::copy_bytes(&length, section.data() + kTextLength, 4);
  std::string text(length, '\0');
  util::copy_bytes(text.data(), section.data() + kTextLength + 4, length);
#if defined(_LIBCPP_VERSION)
  const std::string foreign = text + " 312";  // libstdc++ appends its internal index
#else
  const std::string foreign = text.substr(0, text.rfind(' '));  // libc++ writes one fewer number
#endif
  std::vector<std::byte> rewritten(section.begin(), section.begin() + kTextLength);
  const auto new_length = static_cast<std::uint32_t>(foreign.size());
  const auto* length_bytes = reinterpret_cast<const std::byte*>(&new_length);
  rewritten.insert(rewritten.end(), length_bytes, length_bytes + 4);
  const auto* text_bytes = reinterpret_cast<const std::byte*>(foreign.data());
  rewritten.insert(rewritten.end(), text_bytes, text_bytes + foreign.size());
  rewritten.insert(rewritten.end(), section.begin() + kTextLength + 4 + length, section.end());
  section = std::move(rewritten);

  auto loaded = HnswIndex::from_snapshot(*snapshot);
  ASSERT_FALSE(loaded) << "generator text in the other library's format was accepted";
  EXPECT_EQ(loaded.error().code, ErrorCode::kCorruptData);
  EXPECT_NE(loaded.error().message.find("written by a different one"), std::string::npos)
      << loaded.error().message;
}

}  // namespace
}  // namespace strata
