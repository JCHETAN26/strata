#pragma once

// Internal UTF-8 / UTF-16 / code point helpers for text analysis.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace strata::text {

// Decodes UTF-8 into code points; each invalid byte becomes U+FFFD.
std::u32string decode_utf8(std::string_view s);
std::string encode_utf8(std::u32string_view cps);

std::u16string to_utf16(std::string_view utf8);
std::string to_utf8(std::u16string_view utf16);

// Number of UTF-16 code units (Java String length) for code points.
std::size_t utf16_length(std::u32string_view cps);

}  // namespace strata::text
