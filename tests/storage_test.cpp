#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "strata/collection.hpp"
#include "strata/crc32c.hpp"
#include "strata/snapshot.hpp"
#include "strata/wal.hpp"
#include "test_util.hpp"
#include "util/bytes.hpp"

namespace strata {
namespace {

namespace fs = std::filesystem;

class TempDir : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::temp_directory_path() / (std::string("strata_") + info->test_suite_name() + "_" +
                                        info->name() + "_" + std::to_string(::getpid()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
};

std::vector<char> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

void write_file(const fs::path& path, const std::vector<char>& data) {
  std::ofstream(path, std::ios::binary | std::ios::trunc)
      .write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::span<const std::byte> bytes_of(const std::string& s) { return std::as_bytes(std::span(s)); }

// Deterministic vector for id i, so recovered contents can be checked exactly.
std::vector<float> vector_for(std::size_t i, std::size_t dim) {
  std::vector<float> v(dim);
  for (std::size_t j = 0; j < dim; ++j) {
    v[j] = static_cast<float>(i * 31 + j) * 0.5F;
  }
  return v;
}

// --- CRC32C
// ----------------------------------------------------------------------------------------

TEST(Crc32c, KnownVectors) {
  // Check value from the CRC catalogue (and RFC 3720 appendix B.4).
  EXPECT_EQ(crc32c(bytes_of("123456789")), 0xE3069283U);
  EXPECT_EQ(crc32c({}), 0U);
  const std::string zeros(32, '\0');
  EXPECT_EQ(crc32c(bytes_of(zeros)), 0x8A9136AAU);
}

TEST(Crc32c, IncrementalMatchesWhole) {
  const std::string s = "the quick brown fox jumps over the lazy dog";
  const auto whole = crc32c(bytes_of(s));
  for (std::size_t split = 0; split <= s.size(); ++split) {
    const auto part = crc32c(bytes_of(s.substr(split)), crc32c(bytes_of(s.substr(0, split))));
    ASSERT_EQ(part, whole) << "split " << split;
  }
}

// --- Brute force tombstones
// --------------------------------------------------------------------------

TEST(BruteForceDelete, DeletedVectorsAreNotReturned) {
  auto index = BruteForceIndex::create(2, Metric::kL2);
  ASSERT_TRUE(index);
  for (float x : {0.0F, 1.0F, 2.0F, 3.0F}) {
    ASSERT_TRUE(index->add(std::vector<float>{x, 0}));
  }
  ASSERT_TRUE(index->remove(0));
  ASSERT_TRUE(index->remove(2));
  EXPECT_EQ(index->live_size(), 2U);
  auto result = index->search(std::vector<float>{0, 0}, 10);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->size(), 2U);
  EXPECT_EQ((*result)[0].id, 1U);
  EXPECT_EQ((*result)[1].id, 3U);
}

TEST(BruteForceDelete, RemoveErrors) {
  auto index = BruteForceIndex::create(2, Metric::kL2);
  ASSERT_TRUE(index && index->add(std::vector<float>{1, 1}));
  EXPECT_EQ(index->remove(5).error().code, ErrorCode::kNotFound);
  ASSERT_TRUE(index->remove(0));
  EXPECT_EQ(index->remove(0).error().code, ErrorCode::kNotFound);
  auto result = index->search(std::vector<float>{0, 0}, 3);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->empty());
}

TEST(BruteForceDelete, IdsStayStableAfterDelete) {
  auto index = BruteForceIndex::create(1, Metric::kL2);
  ASSERT_TRUE(index);
  ASSERT_TRUE(index->add(std::vector<float>{0}));
  ASSERT_TRUE(index->remove(0));
  EXPECT_EQ(index->add(std::vector<float>{1}), VectorId{1});
}

// --- Snapshot
// ----------------------------------------------------------------------------------------

using SnapshotTest = TempDir;

TEST_F(SnapshotTest, RoundTrip) {
  Snapshot s;
  s.metric = Metric::kCosine;
  s.vectors = test::random_matrix(13, 5, 1);  // 13: bitmap has a partial byte
  s.deleted = {0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1};
  s.last_lsn = 42;
  ASSERT_TRUE(write_snapshot(dir_ / "s.bin", s));
  auto r = read_snapshot(dir_ / "s.bin");
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(r->metric, Metric::kCosine);
  EXPECT_EQ(r->last_lsn, 42U);
  EXPECT_EQ(r->deleted, s.deleted);
  EXPECT_TRUE(
      std::equal(s.vectors.data().begin(), s.vectors.data().end(), r->vectors.data().begin()));
  EXPECT_FALSE(fs::exists(dir_ / "s.bin.tmp"));
}

TEST_F(SnapshotTest, EmptySnapshot) {
  Snapshot s;
  s.vectors = Matrix<float>(0, 8);
  ASSERT_TRUE(write_snapshot(dir_ / "s.bin", s));
  auto r = read_snapshot(dir_ / "s.bin");
  ASSERT_TRUE(r);
  EXPECT_EQ(r->vectors.rows(), 0U);
  EXPECT_EQ(r->vectors.cols(), 8U);
}

TEST_F(SnapshotTest, EveryFlippedByteIsDetected) {
  Snapshot s;
  s.vectors = test::random_matrix(4, 3, 2);
  s.deleted = {0, 0, 1, 0};
  ASSERT_TRUE(write_snapshot(dir_ / "s.bin", s));
  const auto original = read_file(dir_ / "s.bin");
  for (std::size_t i = 0; i < original.size(); ++i) {
    auto damaged = original;
    damaged[i] = static_cast<char>(damaged[i] ^ 0x5A);
    write_file(dir_ / "s.bin", damaged);
    auto r = read_snapshot(dir_ / "s.bin");
    ASSERT_FALSE(r) << "flip at byte " << i << " not detected";
    EXPECT_EQ(r.error().code, ErrorCode::kCorruptData);
  }
}

// Files written before format version 2 (48-byte header, no byte-order mark or index section)
// still load, as flat snapshots.
TEST_F(SnapshotTest, ReadsVersion1Files) {
  const auto vectors = test::random_matrix(5, 3, 9);
  std::vector<std::byte> buf;
  const auto put = [&](const auto& value) {
    const auto* p = reinterpret_cast<const std::byte*>(&value);
    buf.insert(buf.end(), p, p + sizeof(value));
  };
  for (char c : {'S', 'T', 'R', 'S', 'N', 'P', '\0', '\1'}) {
    put(c);
  }
  put(std::uint32_t{1});                                   // version
  put(static_cast<std::uint32_t>(Metric::kInnerProduct));  // metric
  put(std::uint32_t{3});                                   // dim
  put(std::uint32_t{0});                                   // zero
  put(std::uint64_t{5});                                   // count
  put(std::uint64_t{42});                                  // last_lsn
  put(std::uint64_t{1});                                   // num_deleted
  const auto values = std::as_bytes(vectors.data());
  buf.insert(buf.end(), values.begin(), values.end());
  put(std::uint8_t{0b00100});  // id 2 deleted
  put(crc32c(buf));
  std::vector<char> raw(buf.size());
  util::copy_bytes(raw.data(), buf.data(), buf.size());
  write_file(dir_ / "v1.bin", raw);

  auto r = read_snapshot(dir_ / "v1.bin");
  ASSERT_TRUE(r) << r.error().message;
  EXPECT_EQ(r->index, IndexKind::kFlat);
  EXPECT_EQ(r->metric, Metric::kInnerProduct);
  EXPECT_EQ(r->last_lsn, 42U);
  EXPECT_TRUE(std::ranges::equal(r->vectors.data(), vectors.data()));
  EXPECT_EQ(r->deleted, (std::vector<std::uint8_t>{0, 0, 1, 0, 0}));
  EXPECT_TRUE(r->index_data.empty());
}

TEST_F(SnapshotTest, TruncationIsDetected) {
  Snapshot s;
  s.vectors = test::random_matrix(4, 3, 2);
  s.deleted = {0, 0, 0, 0};
  ASSERT_TRUE(write_snapshot(dir_ / "s.bin", s));
  const auto original = read_file(dir_ / "s.bin");
  for (std::size_t len = 0; len < original.size(); ++len) {
    write_file(dir_ / "s.bin",
               {original.begin(), original.begin() + static_cast<std::ptrdiff_t>(len)});
    ASSERT_FALSE(read_snapshot(dir_ / "s.bin")) << "length " << len;
  }
}

// --- WAL
// -------------------------------------------------------------------------------------------

class WalTest : public TempDir {
 protected:
  static constexpr std::size_t kDim = 4;

  // Opens the WAL and collects everything replayed.
  Expected<std::pair<WriteAheadLog, WalReplayStats>> open(
      std::vector<WalRecord>* out = nullptr, std::vector<std::vector<float>>* vecs = nullptr) {
    return WriteAheadLog::open(path(), kDim, SyncMode::kNone, 1,
                               [&](const WalRecord& r) -> Expected<void> {
                                 if (out != nullptr) {
                                   out->push_back(r);
                                 }
                                 if (vecs != nullptr) {
                                   vecs->emplace_back(r.vector.begin(), r.vector.end());
                                 }
                                 return {};
                               });
  }
  fs::path path() const { return dir_ / "wal.log"; }

  // Writes n inserts (and a delete every 3rd) and returns the file size after each record.
  std::vector<std::size_t> write_records(std::size_t n) {
    auto wal = open();
    EXPECT_TRUE(wal);
    std::vector<std::size_t> ends{fs::file_size(path())};
    for (std::size_t i = 0; i < n; ++i) {
      if (i % 3 == 2) {
        EXPECT_TRUE(wal->first.append_delete(static_cast<VectorId>(i - 1)));
      } else {
        EXPECT_TRUE(wal->first.append_insert(static_cast<VectorId>(i), vector_for(i, kDim)));
      }
      ends.push_back(fs::file_size(path()));
    }
    return ends;
  }
};

TEST_F(WalTest, RoundTrip) {
  write_records(10);
  std::vector<WalRecord> records;
  std::vector<std::vector<float>> vecs;
  auto wal = open(&records, &vecs);
  ASSERT_TRUE(wal) << wal.error().message;
  ASSERT_EQ(records.size(), 10U);
  EXPECT_EQ(wal->second.truncated_bytes, 0U);
  EXPECT_EQ(wal->first.next_lsn(), 11U);
  for (std::size_t i = 0; i < records.size(); ++i) {
    EXPECT_EQ(records[i].lsn, i + 1);
    if (i % 3 == 2) {
      EXPECT_EQ(records[i].type, WalRecordType::kDelete);
      EXPECT_EQ(records[i].id, i - 1);
    } else {
      EXPECT_EQ(records[i].type, WalRecordType::kInsert);
      EXPECT_EQ(records[i].id, i);
      EXPECT_EQ(vecs[i], vector_for(i, kDim));
    }
  }
}

TEST_F(WalTest, AppendAfterReopenContinuesLsn) {
  write_records(3);
  {
    auto wal = open();
    ASSERT_TRUE(wal);
    EXPECT_EQ(*wal->first.append_insert(3, vector_for(3, kDim)), 4U);
  }
  std::vector<WalRecord> records;
  ASSERT_TRUE(open(&records));
  EXPECT_EQ(records.size(), 4U);
}

// Crash mid-append: the file ends partway through the last record. Every truncation point must
// recover all complete records and drop the partial one.
TEST_F(WalTest, TornTailAtEveryOffset) {
  const auto ends = write_records(4);
  const auto full = read_file(path());
  const std::size_t last_start = ends[3];
  for (std::size_t len = last_start + 1; len < full.size(); ++len) {
    write_file(path(), {full.begin(), full.begin() + static_cast<std::ptrdiff_t>(len)});
    std::vector<WalRecord> records;
    auto wal = open(&records);
    ASSERT_TRUE(wal) << "len " << len << ": " << wal.error().message;
    EXPECT_EQ(records.size(), 3U) << "len " << len;
    EXPECT_EQ(wal->second.truncated_bytes, len - last_start);
    EXPECT_EQ(fs::file_size(path()), last_start) << "torn tail not truncated";
  }
}

TEST_F(WalTest, AppendAfterTornTailRecovery) {
  const auto ends = write_records(4);
  fs::resize_file(path(), ends[4] - 5);
  {
    auto wal = open();
    ASSERT_TRUE(wal);
    EXPECT_EQ(*wal->first.append_insert(9, vector_for(9, kDim)), 4U);
  }
  std::vector<WalRecord> records;
  ASSERT_TRUE(open(&records));
  ASSERT_EQ(records.size(), 4U);
  EXPECT_EQ(records[3].id, 9U);
}

TEST_F(WalTest, CorruptionInLastRecordIsTreatedAsTorn) {
  const auto ends = write_records(4);
  auto data = read_file(path());
  data[ends[3] + 25] = static_cast<char>(data[ends[3] + 25] ^ 0xFF);
  write_file(path(), data);
  std::vector<WalRecord> records;
  auto wal = open(&records);
  ASSERT_TRUE(wal);
  EXPECT_EQ(records.size(), 3U);
}

// A bad record with valid data after it means acknowledged writes would be lost: refuse to open.
TEST_F(WalTest, MidLogCorruptionFailsLoudly) {
  const auto ends = write_records(6);
  const auto original = read_file(path());
  for (std::size_t record = 0; record < 5; ++record) {
    for (std::size_t pos : {ends[record], ends[record] + 10, ends[record + 1] - 1}) {
      auto damaged = original;
      damaged[pos] = static_cast<char>(damaged[pos] ^ 0x01);
      write_file(path(), damaged);
      auto wal = open();
      ASSERT_FALSE(wal) << "record " << record << " byte " << pos;
      EXPECT_EQ(wal.error().code, ErrorCode::kCorruptData);
    }
  }
}

// A corrupted length field in the middle of the log can claim the record runs past EOF. That must
// not be mistaken for a torn tail (which would drop the records after it).
TEST_F(WalTest, CorruptLengthMidLogIsNotATornTail) {
  const auto ends = write_records(6);
  const auto original = read_file(path());
  for (std::size_t record = 0; record < 4; ++record) {
    for (std::size_t byte = 4; byte < 8; ++byte) {
      auto damaged = original;
      damaged[ends[record] + byte] = static_cast<char>(0x7F);
      write_file(path(), damaged);
      auto wal = open();
      ASSERT_FALSE(wal) << "record " << record << " length byte " << byte;
      EXPECT_EQ(wal.error().code, ErrorCode::kCorruptData);
    }
  }
}

TEST_F(WalTest, HeaderValidation) {
  write_records(1);
  const auto original = read_file(path());
  for (std::size_t i = 0; i < 28; ++i) {
    auto damaged = original;
    damaged[i] = static_cast<char>(damaged[i] ^ 0x10);
    write_file(path(), damaged);
    ASSERT_FALSE(open()) << "header byte " << i;
  }
  write_file(path(), original);
  auto wrong_dim = WriteAheadLog::open(path(), kDim + 1, SyncMode::kNone, 1,
                                       [](const WalRecord&) -> Expected<void> { return {}; });
  ASSERT_FALSE(wrong_dim);
  EXPECT_EQ(wrong_dim.error().code, ErrorCode::kDimensionMismatch);
}

TEST_F(WalTest, ResetStartsNewLsnRange) {
  write_records(5);
  {
    auto wal = open();
    ASSERT_TRUE(wal);
    ASSERT_TRUE(wal->first.reset(100));
    EXPECT_EQ(*wal->first.append_insert(0, vector_for(0, kDim)), 100U);
  }
  std::vector<WalRecord> records;
  auto wal = open(&records);
  ASSERT_TRUE(wal);
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records[0].lsn, 100U);
}

TEST_F(WalTest, ApplyErrorAbortsOpen) {
  write_records(3);
  auto wal = WriteAheadLog::open(path(), kDim, SyncMode::kNone, 1,
                                 [](const WalRecord& r) -> Expected<void> {
                                   if (r.lsn == 2) {
                                     return make_error(ErrorCode::kCorruptData, "nope");
                                   }
                                   return {};
                                 });
  EXPECT_FALSE(wal);
}

// --- Collection
// ------------------------------------------------------------------------------------

using CollectionTest = TempDir;

TEST_F(CollectionTest, InsertSearchReopen) {
  {
    auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
    ASSERT_TRUE(c) << c.error().message;
    for (std::size_t i = 0; i < 20; ++i) {
      ASSERT_EQ(c->insert(vector_for(i, 3)), VectorId(i));
    }
    ASSERT_TRUE(c->remove(4));
  }
  auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
  ASSERT_TRUE(c);
  EXPECT_EQ(c->size(), 20U);
  EXPECT_EQ(c->live_size(), 19U);
  EXPECT_EQ(c->recovery().wal_records_replayed, 21U);
  EXPECT_FALSE(c->get(4).has_value());
  EXPECT_EQ(*c->get(7), vector_for(7, 3));
  auto r = c->search(vector_for(4, 3), 1);
  ASSERT_TRUE(r);
  EXPECT_NE((*r)[0].id, 4U);
}

TEST_F(CollectionTest, CheckpointThenMoreWrites) {
  {
    auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
    ASSERT_TRUE(c);
    for (std::size_t i = 0; i < 10; ++i) {
      ASSERT_TRUE(c->insert(vector_for(i, 3)));
    }
    ASSERT_TRUE(c->remove(2));
    ASSERT_TRUE(c->checkpoint());
    EXPECT_EQ(c->wal_size_bytes(), 32U);  // header only
    for (std::size_t i = 10; i < 15; ++i) {
      ASSERT_TRUE(c->insert(vector_for(i, 3)));
    }
  }
  auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
  ASSERT_TRUE(c);
  EXPECT_TRUE(c->recovery().loaded_snapshot);
  EXPECT_EQ(c->recovery().wal_records_replayed, 5U);
  EXPECT_EQ(c->size(), 15U);
  EXPECT_FALSE(c->get(2).has_value());
  for (std::size_t i = 0; i < 15; ++i) {
    if (i != 2) {
      EXPECT_EQ(*c->get(static_cast<VectorId>(i)), vector_for(i, 3)) << i;
    }
  }
}

// Crash after the snapshot is written but before the WAL is reset: the old WAL still holds records
// the snapshot includes. They must be skipped, not applied twice.
TEST_F(CollectionTest, CrashBetweenSnapshotAndWalReset) {
  {
    auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
    ASSERT_TRUE(c);
    for (std::size_t i = 0; i < 6; ++i) {
      ASSERT_TRUE(c->insert(vector_for(i, 3)));
    }
    const auto old_wal = read_file(dir_ / "wal.log");
    ASSERT_TRUE(c->checkpoint());
    write_file(dir_ / "wal.log", old_wal);  // as if the reset never happened
  }
  auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
  ASSERT_TRUE(c) << c.error().message;
  EXPECT_EQ(c->size(), 6U);
  EXPECT_EQ(c->recovery().wal_records_skipped, 6U);
  // New writes continue after the old records and survive another reopen.
  ASSERT_EQ(c->insert(vector_for(6, 3)), VectorId{6});
  auto again = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
  ASSERT_TRUE(again);
}

TEST_F(CollectionTest, MismatchedDimensionOrMetric) {
  {
    auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
    ASSERT_TRUE(c && c->insert(vector_for(0, 3)) && c->checkpoint());
  }
  EXPECT_FALSE(Collection::open(dir_, 4, Metric::kL2));
  EXPECT_FALSE(Collection::open(dir_, 3, Metric::kCosine));
}

TEST_F(CollectionTest, InvalidWritesAreNotLogged) {
  auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
  ASSERT_TRUE(c);
  const auto before = c->wal_size_bytes();
  EXPECT_FALSE(c->insert(std::vector<float>{1, 2}));
  EXPECT_FALSE(c->remove(0));
  EXPECT_EQ(c->wal_size_bytes(), before);
}

// Readers search while a writer inserts and deletes; TSan (tsan preset) checks the locking.
TEST_F(CollectionTest, ConcurrentSearchesDuringWrites) {
  auto c = Collection::open(dir_, 3, Metric::kL2, {.sync = SyncMode::kNone});
  ASSERT_TRUE(c);
  std::atomic<bool> done{false};
  std::atomic<std::size_t> searches{0};
  std::vector<std::jthread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&] {
      while (!done.load()) {
        auto r = c->search(vector_for(1, 3), 5);
        ASSERT_TRUE(r);
        ASSERT_LE(r->size(), 5U);
        searches.fetch_add(1);
      }
    });
  }
  for (std::size_t i = 0; i < 300; ++i) {
    ASSERT_TRUE(c->insert(vector_for(i, 3)));
    if (i % 10 == 9) {
      ASSERT_TRUE(c->remove(static_cast<VectorId>(i - 5)));
    }
    if (i % 100 == 99) {
      ASSERT_TRUE(c->checkpoint());
    }
  }
  done = true;
  readers.clear();
  EXPECT_GT(searches.load(), 0U);
  EXPECT_EQ(c->live_size(), 270U);
}

}  // namespace
}  // namespace strata
