#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata {

// Fixed-size bitset over vector ids, 64 ids per word.
// Thread safety: concurrent const access is safe; mutation requires exclusive access.
class Bitset {
 public:
  Bitset() = default;
  explicit Bitset(std::size_t size) : size_(size), words_((size + 63) / 64, 0) {}

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool test(std::size_t i) const noexcept {
    return ((words_[i / 64] >> (i % 64)) & 1U) != 0;
  }
  void set(std::size_t i) noexcept { words_[i / 64] |= std::uint64_t{1} << (i % 64); }
  void reset(std::size_t i) noexcept { words_[i / 64] &= ~(std::uint64_t{1} << (i % 64)); }

  [[nodiscard]] std::size_t count() const noexcept {
    std::size_t n = 0;
    for (std::uint64_t w : words_) {
      n += static_cast<std::size_t>(std::popcount(w));
    }
    return n;
  }

  // Calls fn(i) for every set bit in increasing order. Skips empty words, so the cost is
  // O(size / 64 + count) rather than O(size).
  template <typename Fn>
  void for_each_set(Fn&& fn) const {
    for (std::size_t w = 0; w < words_.size(); ++w) {
      std::uint64_t word = words_[w];
      while (word != 0) {
        const auto bit = static_cast<std::size_t>(std::countr_zero(word));
        fn(w * 64 + bit);
        word &= word - 1;  // clear lowest set bit
      }
    }
  }

  [[nodiscard]] std::vector<std::uint64_t>& words() noexcept { return words_; }
  [[nodiscard]] const std::vector<std::uint64_t>& words() const noexcept { return words_; }

 private:
  std::size_t size_ = 0;
  std::vector<std::uint64_t> words_;
};

}  // namespace strata
