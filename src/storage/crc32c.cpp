#include "strata/crc32c.hpp"

#include <array>

namespace strata {

namespace {

constexpr std::uint32_t kPolynomial = 0x82F63B78U;  // reversed Castagnoli polynomial

constexpr std::array<std::uint32_t, 256> make_table() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int bit = 0; bit < 8; ++bit) {
      c = (c & 1U) != 0 ? (c >> 1U) ^ kPolynomial : c >> 1U;
    }
    table[i] = c;
  }
  return table;
}

constexpr auto kTable = make_table();

}  // namespace

std::uint32_t crc32c(std::span<const std::byte> data, std::uint32_t crc) noexcept {
  crc = ~crc;
  for (std::byte b : data) {
    crc = kTable[(crc ^ static_cast<std::uint32_t>(b)) & 0xFFU] ^ (crc >> 8U);
  }
  return ~crc;
}

}  // namespace strata
