// Crash-recovery tests: a child process writes to a Collection as fast as it can and reports each
// acknowledged write through a pipe; the parent SIGKILLs it at a random moment (possibly inside a
// WAL append, between append and ack, or mid-checkpoint), then reopens the collection and checks
// that every acknowledged write is present and every recovered vector is intact.
//
// SIGKILL leaves the OS page cache intact, so this tests the software protocol (record framing,
// torn-tail handling, checkpoint ordering), including with SyncMode::kNone. Power-loss durability
// additionally depends on fsync reaching stable storage, which a process kill cannot simulate.

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <ostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "strata/collection.hpp"
#include "test_util.hpp"

namespace strata {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kDim = 8;
constexpr std::uint32_t kDeleteFlag = 0x80000000U;

std::vector<float> vector_for(std::size_t i) {
  std::vector<float> v(kDim);
  for (std::size_t j = 0; j < kDim; ++j) {
    v[j] = static_cast<float>((i * 131) + j) * 0.25F;
  }
  return v;
}

// Child process body. Never returns. Reports each acknowledged write as a u32 on `ack_fd`:
// the id for an insert, id | kDeleteFlag for a delete.
[[noreturn]] void writer(const fs::path& dir, SyncMode sync, std::size_t checkpoint_every,
                         int ack_fd) {
  auto collection = Collection::open(dir, kDim, Metric::kL2, {.sync = sync});
  if (!collection) {
    _exit(2);
  }
  for (std::size_t i = collection->size();; ++i) {
    auto id = collection->insert(vector_for(i));
    if (!id || *id != i) {
      _exit(3);
    }
    std::uint32_t msg = *id;
    if (::write(ack_fd, &msg, sizeof(msg)) != sizeof(msg)) {
      _exit(4);
    }
    if (i % 5 == 4) {  // delete an earlier vector now and then
      const auto victim = static_cast<VectorId>(i - 2);
      if (!collection->remove(victim)) {
        _exit(5);
      }
      msg = victim | kDeleteFlag;
      if (::write(ack_fd, &msg, sizeof(msg)) != sizeof(msg)) {
        _exit(4);
      }
    }
    if (checkpoint_every != 0 && i % checkpoint_every == checkpoint_every - 1) {
      if (!collection->checkpoint()) {
        _exit(6);
      }
    }
  }
}

struct Acks {
  std::set<VectorId> inserted;
  std::set<VectorId> deleted;
};

// Runs one writer, kills it after at least `min_acks` acknowledgments, and returns every ack it
// managed to send (including ones that arrived after the kill decision).
Acks run_and_kill(const fs::path& dir, SyncMode sync, std::size_t checkpoint_every,
                  std::size_t min_acks) {
  std::array<int, 2> fds{};
  EXPECT_EQ(::pipe(fds.data()), 0);
  const pid_t pid = ::fork();
  if (pid == 0) {
    ::close(fds[0]);
    writer(dir, sync, checkpoint_every, fds[1]);
  }
  ::close(fds[1]);

  Acks acks;
  auto record = [&](std::uint32_t msg) {
    if ((msg & kDeleteFlag) != 0) {
      acks.deleted.insert(msg & ~kDeleteFlag);
    } else {
      acks.inserted.insert(msg);
    }
  };
  std::uint32_t msg = 0;
  std::size_t count = 0;
  while (count < min_acks && ::read(fds[0], &msg, sizeof(msg)) == sizeof(msg)) {
    record(msg);
    ++count;
  }
  ::kill(pid, SIGKILL);
  int status = 0;
  ::waitpid(pid, &status, 0);
  EXPECT_TRUE(WIFSIGNALED(status))
      << "writer exited on its own with status " << WEXITSTATUS(status);
  // Drain acks written before the kill landed; those writes were acknowledged too.
  while (::read(fds[0], &msg, sizeof(msg)) == sizeof(msg)) {
    record(msg);
  }
  ::close(fds[0]);
  return acks;
}

struct CrashCase {
  std::string name;
  SyncMode sync;
  std::size_t checkpoint_every;  // 0 = never
  int rounds;
};

// Printed as its name ("wal_only"), not as bytes of the struct (which include a heap pointer).
void PrintTo(const CrashCase& c, std::ostream* os) { *os << c.name; }

class CrashRecovery : public ::testing::TestWithParam<CrashCase> {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() /
           ("strata_crash_" + GetParam().name + "_" + std::to_string(::getpid()));
    fs::remove_all(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  fs::path dir_;
};

TEST_P(CrashRecovery, NoAcknowledgedWriteIsLost) {
  const auto& param = GetParam();
  std::mt19937 rng(20260925);
  std::uniform_int_distribution<std::size_t> kill_after(1, 400);
  Acks all;  // accumulated across rounds: each round reopens and continues the same collection

  for (int round = 0; round < param.rounds; ++round) {
    const Acks acks = run_and_kill(dir_, param.sync, param.checkpoint_every, kill_after(rng));
    all.inserted.insert(acks.inserted.begin(), acks.inserted.end());
    all.deleted.insert(acks.deleted.begin(), acks.deleted.end());

    auto c = Collection::open(dir_, kDim, Metric::kL2, {.sync = param.sync});
    ASSERT_TRUE(c) << "round " << round << ": recovery failed: " << c.error().message;
    SCOPED_TRACE("round " + std::to_string(round));

    // Every acknowledged insert is present with exactly the right contents, unless its delete was
    // also acknowledged.
    for (VectorId id : all.inserted) {
      ASSERT_LT(id, c->size()) << "acknowledged insert " << id << " lost";
      const auto stored = c->get(id);
      // The writer deletes id i - 2 after inserting i when i % 5 == 4, i.e. ids with id % 5 == 2.
      // Such a delete may be durable without its ack having arrived, so absence is allowed.
      const bool delete_possible = id % 5 == 2 && id + 2 < c->size();
      if (all.deleted.contains(id)) {
        EXPECT_FALSE(stored.has_value()) << "acknowledged delete of " << id << " lost";
      } else if (!stored.has_value()) {
        EXPECT_TRUE(delete_possible) << "vector " << id << " missing";
      } else {
        EXPECT_EQ(*stored, vector_for(id)) << "vector " << id << " corrupted";
      }
    }
    // Writes that became durable but whose ack never arrived are allowed, but must be intact.
    for (VectorId id = 0; id < c->size(); ++id) {
      if (const auto stored = c->get(id); stored.has_value()) {
        EXPECT_EQ(*stored, vector_for(id)) << "unacknowledged vector " << id << " corrupted";
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Modes, CrashRecovery,
                         ::testing::Values(CrashCase{"wal_only", SyncMode::kNone, 0, 15},
                                           CrashCase{"frequent_checkpoints", SyncMode::kNone, 7,
                                                     15},
                                           CrashCase{"rare_checkpoints", SyncMode::kNone, 97, 15},
                                           CrashCase{"fsync", SyncMode::kFsync, 13, 4}),
                         test::PrintedName{});

}  // namespace
}  // namespace strata
