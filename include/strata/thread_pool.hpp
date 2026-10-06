#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata {

// Fixed-size pool for data-parallel loops (batch search, parallel index build).
//
// parallel_for(n, fn) runs fn(i) for every i in [0, n) and returns when all calls have finished.
// Work is handed out in chunks through an atomic counter (dynamic scheduling), so uneven
// per-item cost (e.g. graph searches of different lengths) still balances. The calling thread
// participates, so a pool of size 1 has no worker threads and runs everything inline.
//
// fn must not throw (hot paths return errors instead) and must be safe to call concurrently for
// different i.
//
// Thread safety: parallel_for may be called from several threads; calls are serialized.
// Destruction must not overlap with a parallel_for call.
class ThreadPool {
 public:
  // num_threads == 0 means std::thread::hardware_concurrency().
  explicit ThreadPool(std::size_t num_threads = 0);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // Total threads that execute work, including the caller.
  [[nodiscard]] std::size_t size() const noexcept { return workers_.size() + 1; }

  // chunk == 0 picks a chunk size giving each thread several chunks.
  void parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn,
                    std::size_t chunk = 0);

 private:
  void worker_loop();
  void run_chunks();

  // Joined explicitly in ~ThreadPool, before the synchronization members below are destroyed.
  // std::thread, not std::jthread: Xcode 16's libc++ (Apple clang 16) has no jthread.
  std::vector<std::thread> workers_;

  std::mutex call_mutex_;  // serializes parallel_for calls

  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  std::size_t generation_ = 0;  // bumped per job so workers don't rerun a finished one
  std::size_t active_workers_ = 0;
  bool stop_ = false;

  // Current job. Written under mutex_ before generation_ is bumped; read by workers after they
  // observe the new generation under mutex_, so no extra synchronization is needed.
  const std::function<void(std::size_t)>* fn_ = nullptr;
  std::size_t n_ = 0;
  std::size_t chunk_ = 1;
  std::atomic<std::size_t> next_{0};
};

}  // namespace strata
