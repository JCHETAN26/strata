#include <utf8proc.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "strata/text.hpp"
#include "text_util.hpp"

namespace strata {

// --- UTF helpers ----------

namespace text {

std::u32string decode_utf8(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  const auto* p = reinterpret_cast<const utf8proc_uint8_t*>(s.data());
  auto remaining = static_cast<utf8proc_ssize_t>(s.size());
  while (remaining > 0) {
    utf8proc_int32_t cp = 0;
    const utf8proc_ssize_t n = utf8proc_iterate(p, remaining, &cp);
    if (n <= 0) {
      out.push_back(U'�');
      ++p;
      --remaining;
    } else {
      out.push_back(static_cast<char32_t>(cp));
      p += n;
      remaining -= n;
    }
  }
  return out;
}

std::string encode_utf8(std::u32string_view cps) {
  std::string out;
  out.reserve(cps.size());
  std::array<utf8proc_uint8_t, 4> buf{};
  for (char32_t cp : cps) {
    const auto n = utf8proc_encode_char(static_cast<utf8proc_int32_t>(cp), buf.data());
    out.append(reinterpret_cast<const char*>(buf.data()), static_cast<std::size_t>(n));
  }
  return out;
}

std::u16string to_utf16(std::string_view utf8) {
  std::u16string out;
  for (char32_t cp : decode_utf8(utf8)) {
    if (cp >= 0x10000) {
      const char32_t v = cp - 0x10000;
      out.push_back(static_cast<char16_t>(0xD800 + (v >> 10U)));
      out.push_back(static_cast<char16_t>(0xDC00 + (v & 0x3FFU)));
    } else {
      out.push_back(static_cast<char16_t>(cp));
    }
  }
  return out;
}

std::string to_utf8(std::u16string_view utf16) {
  std::u32string cps;
  for (std::size_t i = 0; i < utf16.size(); ++i) {
    const char32_t u = utf16[i];
    if (u >= 0xD800 && u <= 0xDBFF && i + 1 < utf16.size() && utf16[i + 1] >= 0xDC00 &&
        utf16[i + 1] <= 0xDFFF) {
      cps.push_back(0x10000 + ((u - 0xD800) << 10U) + (utf16[i + 1] - 0xDC00));
      ++i;
    } else {
      cps.push_back(u);
    }
  }
  return encode_utf8(cps);
}

std::size_t utf16_length(std::u32string_view cps) {
  std::size_t n = 0;
  for (char32_t cp : cps) {
    n += cp >= 0x10000 ? 2 : 1;
  }
  return n;
}

}  // namespace text

// --- Character classes ----------

namespace {

#include "emoji_table.inc"

bool is_emoji(char32_t cp) {
  const auto it =
      std::ranges::upper_bound(kEmojiRanges, cp, {}, &std::pair<char32_t, char32_t>::first);
  return it != kEmojiRanges.begin() && cp <= std::prev(it)->second;
}

bool is_regional_indicator(char32_t cp) { return cp >= 0x1F1E6 && cp <= 0x1F1FF; }
bool is_emoji_modifier(char32_t cp) { return cp >= 0x1F3FB && cp <= 0x1F3FF; }

// UAX #29 Word_Break classes, as far as this tokenizer distinguishes them.
enum class Wb {
  kALetter,       // letters (general category L*, Nl), including Hangul and Southeast Asian
  kNumeric,       // decimal digits (Nd)
  kKatakana,      // joins with Katakana
  kExtendNumLet,  // '_' and connector punctuation: joins letters/digits
  kMidLetter,     // joins letters only: ':' U+00B7 ...
  kMidNum,        // joins digits only: ',' ';' ...
  kMidNumLet,     // joins letters or digits: '.' U+2019 ...
  kSingleQuote,   // '\'': like MidNumLet
  kExtend,        // combining marks, ZWJ, format characters: attach to the previous character
  kIdeographic,   // Han, Hiragana: each character is its own token
  kEmoji,         // Emoji=Yes symbols (not letters): each emoji sequence is its own token
  kOther,         // separators, punctuation, symbols
};

bool in(char32_t cp, std::initializer_list<char32_t> set) {
  for (char32_t c : set) {
    if (c == cp) {
      return true;
    }
  }
  return false;
}

Wb word_break(char32_t cp) {
  if (cp == U'\'') {
    return Wb::kSingleQuote;
  }
  if (in(cp, {0x2E, 0x2018, 0x2019, 0x2024, 0xFE52, 0xFF07, 0xFF0E})) {
    return Wb::kMidNumLet;
  }
  if (in(cp, {0x3A, 0xB7, 0x387, 0x55F, 0x5F4, 0x2027, 0xFE13, 0xFE55, 0xFF1A})) {
    return Wb::kMidLetter;
  }
  if (in(cp, {0x2C, 0x3B, 0x37E, 0x589, 0x60C, 0x60D, 0x66C, 0x7F8, 0x2044, 0xFE10, 0xFE14, 0xFE50,
              0xFE54, 0xFF0C, 0xFF1B})) {
    return Wb::kMidNum;
  }
  if (in(cp,
         {0x5F, 0x202F, 0x203F, 0x2040, 0x2054, 0xFE33, 0xFE34, 0xFE4D, 0xFE4E, 0xFE4F, 0xFF3F})) {
    return Wb::kExtendNumLet;
  }
  if (cp == 0x200D) {
    return Wb::kExtend;  // ZWJ
  }
  // Hiragana and Han ideographs: one token per character (Lucene's HIRAGANA / IDEOGRAPHIC).
  if ((cp >= 0x3041 && cp <= 0x309F) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
      (cp >= 0x20000 && cp <= 0x2FA1F) || (cp >= 0x3005 && cp <= 0x3007) ||
      (cp >= 0x3021 && cp <= 0x3029)) {
    return Wb::kIdeographic;
  }
  if ((cp >= 0x30A1 && cp <= 0x30FA) || (cp >= 0x30FC && cp <= 0x30FF) ||
      (cp >= 0x31F0 && cp <= 0x31FF) || (cp >= 0xFF66 && cp <= 0xFF9D)) {
    return Wb::kKatakana;
  }
  const auto category = utf8proc_category(static_cast<utf8proc_int32_t>(cp));
  if (is_emoji(cp) && !(category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO)) {
    return Wb::kEmoji;  // e.g. (c) (R) TM <-> and pictographs; Emoji=Yes letters stay letters
  }
  switch (category) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
    case UTF8PROC_CATEGORY_NL:
      return Wb::kALetter;
    case UTF8PROC_CATEGORY_ND:
      return Wb::kNumeric;
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
    case UTF8PROC_CATEGORY_CF:
      return Wb::kExtend;
    default:
      return Wb::kOther;
  }
}

bool is_word_core(Wb c) { return c == Wb::kALetter || c == Wb::kNumeric || c == Wb::kKatakana; }

// UAX #29 rules WB5, WB8-WB10, WB13, WB13a/b: may `prev` and `next` be adjacent in one word?
bool joins(Wb prev, Wb next) {
  const bool prev_word = is_word_core(prev);
  const bool next_word = is_word_core(next);
  if (prev == Wb::kKatakana || next == Wb::kKatakana) {
    return (prev == Wb::kKatakana && next == Wb::kKatakana) ||
           (prev == Wb::kExtendNumLet && next == Wb::kKatakana) ||
           (prev == Wb::kKatakana && next == Wb::kExtendNumLet);
  }
  if (prev_word && next_word) {
    return true;  // letters and digits join each other
  }
  return (prev_word || prev == Wb::kExtendNumLet) && (next_word || next == Wb::kExtendNumLet) &&
         (prev == Wb::kExtendNumLet || next == Wb::kExtendNumLet);
}

// UAX #29 WB6/7 (letters) and WB11/12 (digits): may `mid` sit between prev and next in a word?
bool mid_joins(Wb prev, Wb mid, Wb next) {
  if (prev == Wb::kALetter && next == Wb::kALetter) {
    return mid == Wb::kMidLetter || mid == Wb::kMidNumLet || mid == Wb::kSingleQuote;
  }
  if (prev == Wb::kNumeric && next == Wb::kNumeric) {
    return mid == Wb::kMidNum || mid == Wb::kMidNumLet || mid == Wb::kSingleQuote;
  }
  return false;
}

// A token being built, with the Word_Break class of its last non-Extend character.
struct Token {
  std::u32string chars;
  Wb last = Wb::kOther;
  bool has_word_char = false;  // Lucene emits only segments containing a letter or digit
};

void emit(Token& token, std::size_t max_length, std::vector<std::u32string>& out) {
  if (token.has_word_char && !token.chars.empty()) {
    // Split overlong tokens every max_length UTF-16 units, as Lucene does.
    std::u32string piece;
    std::size_t units = 0;
    for (char32_t cp : token.chars) {
      const std::size_t w = cp >= 0x10000 ? 2 : 1;
      if (units + w > max_length && !piece.empty()) {
        out.push_back(std::move(piece));
        piece.clear();
        units = 0;
      }
      piece.push_back(cp);
      units += w;
    }
    if (!piece.empty()) {
      out.push_back(std::move(piece));
    }
  }
  token = Token{};
}

std::vector<std::u32string> tokenize_unicode_words(const std::u32string& text,
                                                   std::size_t max_length) {
  std::vector<std::u32string> out;
  Token token;
  const std::size_t n = text.size();
  for (std::size_t i = 0; i < n; ++i) {
    const char32_t cp = text[i];
    const Wb c = word_break(cp);
    if (c == Wb::kExtend) {
      if (!token.chars.empty()) {
        token.chars.push_back(cp);  // WB4: marks attach to the preceding character
      }
      continue;
    }
    if (c == Wb::kIdeographic) {
      emit(token, max_length, out);
      out.emplace_back(1, cp);
      continue;
    }
    if (c == Wb::kEmoji) {
      // Lucene emits emoji as <EMOJI> tokens: the emoji, its modifiers / variation selectors,
      // ZWJ-joined emoji, and regional-indicator pairs (flags) form one token.
      emit(token, max_length, out);
      std::u32string emoji(1, cp);
      std::size_t j = i + 1;
      if (is_regional_indicator(cp) && j < n && is_regional_indicator(text[j])) {
        emoji.push_back(text[j++]);
      }
      while (j < n) {
        const char32_t next = text[j];
        if (next == 0x200D && j + 1 < n && word_break(text[j + 1]) == Wb::kEmoji) {
          emoji.push_back(next);
          emoji.push_back(text[j + 1]);
          j += 2;
        } else if ((word_break(next) == Wb::kExtend && next != 0x200D) || is_emoji_modifier(next)) {
          emoji.push_back(next);
          ++j;
        } else {
          break;
        }
      }
      out.push_back(std::move(emoji));
      i = j - 1;
      continue;
    }
    if (is_word_core(c) || c == Wb::kExtendNumLet) {
      if (!token.chars.empty() && !joins(token.last, c)) {
        emit(token, max_length, out);
      }
      token.chars.push_back(cp);
      token.last = c;
      token.has_word_char = token.has_word_char || is_word_core(c);
      continue;
    }
    if ((c == Wb::kMidLetter || c == Wb::kMidNum || c == Wb::kMidNumLet || c == Wb::kSingleQuote) &&
        !token.chars.empty()) {
      // Look past marks for the next real character.
      std::size_t j = i + 1;
      while (j < n && word_break(text[j]) == Wb::kExtend) {
        ++j;
      }
      if (j < n && mid_joins(token.last, c, word_break(text[j]))) {
        token.chars.push_back(cp);  // `last` stays: the next character decides the rest
        continue;
      }
    }
    emit(token, max_length, out);
  }
  emit(token, max_length, out);
  return out;
}

std::vector<std::u32string> tokenize_plain(const std::u32string& text, std::size_t max_length) {
  std::vector<std::u32string> out;
  Token token;
  for (char32_t cp : text) {
    const auto category = utf8proc_category(static_cast<utf8proc_int32_t>(cp));
    const bool letter_or_digit =
        (category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO) ||
        (category >= UTF8PROC_CATEGORY_ND && category <= UTF8PROC_CATEGORY_NO);
    const bool mark = category >= UTF8PROC_CATEGORY_MN && category <= UTF8PROC_CATEGORY_ME;
    if (letter_or_digit || (mark && !token.chars.empty())) {
      token.chars.push_back(cp);
      token.has_word_char = true;
    } else {
      emit(token, max_length, out);
    }
  }
  emit(token, max_length, out);
  return out;
}

constexpr std::array<std::string_view, 33> kLuceneEnglishStopwords = {
    "a",   "an",    "and",  "are",   "as",    "at",   "be",   "but", "by",  "for",  "if",
    "in",  "into",  "is",   "it",    "no",    "not",  "of",   "on",  "or",  "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to",  "was", "will", "with"};

}  // namespace

bool is_lucene_english_stopword(std::string_view word) noexcept {
  for (std::string_view s : kLuceneEnglishStopwords) {
    if (s == word) {
      return true;
    }
  }
  return false;
}

AnalyzerConfig AnalyzerConfig::plain() {
  return {.tokenizer = Tokenizer::kPlain,
          .english_possessive = false,
          .lowercase = true,
          .stopwords = Stopwords::kNone,
          .stemmer = Stemmer::kNone};
}

AnalyzerConfig AnalyzerConfig::anserini_english() {
  return {.tokenizer = Tokenizer::kUnicodeWords,
          .english_possessive = true,
          .lowercase = true,
          .stopwords = Stopwords::kLuceneEnglish,
          .stemmer = Stemmer::kPorter};
}

Analyzer::Analyzer(AnalyzerConfig config) : config_(config) {}

std::vector<std::string> Analyzer::analyze(std::string_view text) const {
  const std::u32string cps = text::decode_utf8(text);
  std::vector<std::u32string> tokens = config_.tokenizer == Tokenizer::kPlain
                                           ? tokenize_plain(cps, config_.max_token_length)
                                           : tokenize_unicode_words(cps, config_.max_token_length);
  std::vector<std::string> out;
  out.reserve(tokens.size());
  for (auto& token : tokens) {
    // Lucene EnglishPossessiveFilter: drop a trailing 's (', U+2019, U+FF07).
    if (config_.english_possessive && token.size() >= 2) {
      const char32_t quote = token[token.size() - 2];
      const char32_t s = token.back();
      if ((quote == U'\'' || quote == 0x2019 || quote == 0xFF07) && (s == U's' || s == U'S')) {
        token.resize(token.size() - 2);
      }
    }
    if (config_.lowercase) {
      for (char32_t& cp : token) {
        cp = static_cast<char32_t>(utf8proc_tolower(static_cast<utf8proc_int32_t>(cp)));
      }
    }
    if (token.empty()) {
      continue;
    }
    std::string word = text::encode_utf8(token);
    if (config_.stopwords == Stopwords::kLuceneEnglish && is_lucene_english_stopword(word)) {
      continue;
    }
    if (config_.stemmer == Stemmer::kPorter) {
      word = porter_stem(word);
    }
    out.push_back(std::move(word));
  }
  return out;
}

}  // namespace strata
