#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata {

// CRC-32C (Castagnoli), the checksum used by iSCSI, ext4 metadata, and RocksDB/LevelDB logs.
// Detects all burst errors up to 32 bits, so a torn or partially zeroed record is always caught.
// Table-driven software implementation (~1 byte/cycle); WAL throughput is bound by fsync, not this.
// Pass the previous result as `crc` to checksum data in pieces.
// Thread safety: pure function.
[[nodiscard]] std::uint32_t crc32c(std::span<const std::byte> data, std::uint32_t crc = 0) noexcept;

}  // namespace strata
