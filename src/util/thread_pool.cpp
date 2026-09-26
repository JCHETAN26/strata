#include "strata/thread_pool.hpp"

#include <algorithm>

namespace strata {

ThreadPool::ThreadPool(std::size_t num_threads) {
  if (num_threads == 0) {
    num_threads = std::max(1U, std::thread::hardware_concurrency());
  }
  workers_.reserve(num_threads - 1);
  for (std::size_t i = 0; i + 1 < num_threads; ++i) {
    workers_.emplace_back([this] { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    const std::lock_guard lock(mutex_);
    stop_ = true;
  }
  work_ready_.notify_all();
  // Join now, while mutex_ and the condition variables are still alive. Leaving it to member
  // destruction would join last (workers_ is declared first), after the state they use is gone.
  workers_.clear();
}

void ThreadPool::run_chunks() {
  while (true) {
    const std::size_t begin = next_.fetch_add(chunk_, std::memory_order_relaxed);
    if (begin >= n_) {
      return;
    }
    const std::size_t end = std::min(begin + chunk_, n_);
    for (std::size_t i = begin; i < end; ++i) {
      (*fn_)(i);
    }
  }
}

void ThreadPool::worker_loop() {
  std::size_t seen_generation = 0;
  while (true) {
    {
      std::unique_lock lock(mutex_);
      work_ready_.wait(lock, [&] { return stop_ || generation_ != seen_generation; });
      if (stop_) {
        return;
      }
      seen_generation = generation_;
    }
    run_chunks();
    {
      const std::lock_guard lock(mutex_);
      --active_workers_;
    }
    work_done_.notify_one();
  }
}

void ThreadPool::parallel_for(std::size_t n, const std::function<void(std::size_t)>& fn,
                              std::size_t chunk) {
  if (n == 0) {
    return;
  }
  const std::lock_guard call_lock(call_mutex_);
  if (chunk == 0) {
    // ~4 chunks per thread: enough to rebalance, few enough that the atomic isn't contended.
    chunk = std::max<std::size_t>(1, n / (size() * 4));
  }
  if (workers_.empty() || n <= chunk) {
    for (std::size_t i = 0; i < n; ++i) {
      fn(i);
    }
    return;
  }
  {
    const std::lock_guard lock(mutex_);
    fn_ = &fn;
    n_ = n;
    chunk_ = chunk;
    next_.store(0, std::memory_order_relaxed);
    active_workers_ = workers_.size();
    ++generation_;
  }
  work_ready_.notify_all();
  run_chunks();
  std::unique_lock lock(mutex_);
  work_done_.wait(lock, [&] { return active_workers_ == 0; });
  fn_ = nullptr;
}

}  // namespace strata
