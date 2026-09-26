#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

// Text analysis for BM25: tokenizer -> possessive filter -> lowercase -> stopwords -> stemmer,
// in that order (the order Anserini's DefaultEnglishAnalyzer uses).
//
// Input is UTF-8. Invalid byte sequences are treated as U+FFFD (a separator). Output tokens are
// UTF-8.

enum class Tokenizer {
  // Maximal runs of letters and digits (Unicode general categories L*, N*, plus combining marks
  // attached to them); everything else separates. Simple and predictable.
  kPlain,
  // Unicode word segmentation following UAX #29 word-boundary rules, as Lucene's
  // StandardTokenizer applies them: letters and digits join across single mid-word
  // punctuation ("don't", "u.s.a", "3.14", "1,000", "x.com"), underscores join words, each Han
  // ideograph and Hiragana character is its own token, punctuation-only segments are dropped.
  // Word_Break classes are derived from the Unicode general category (via utf8proc) plus the
  // explicit mid-word punctuation lists of UAX #29. Emoji (Unicode Emoji=Yes symbols, from
  // src/text/emoji_table.inc) are their own tokens, with modifiers, ZWJ sequences, and flags
  // kept together, as Lucene emits them. Not implemented: the Hebrew-letter rules.
  kUnicodeWords,
};

enum class Stopwords {
  kNone,
  // Lucene's EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (33 words), Anserini's default.
  kLuceneEnglish,
};

enum class Stemmer {
  kNone,
  // Porter (1980) as implemented in Lucene's PorterStemmer (Porter's reference code, including
  // his later departures from the paper: -bli -> -ble, -logi -> -log).
  kPorter,
};

struct AnalyzerConfig {
  Tokenizer tokenizer = Tokenizer::kUnicodeWords;
  // Strip a trailing 's (', U+2019, or U+FF07 followed by s/S), before lowercasing.
  bool english_possessive = false;
  bool lowercase = true;
  Stopwords stopwords = Stopwords::kNone;
  Stemmer stemmer = Stemmer::kNone;
  // Longer tokens are split into chunks of this many UTF-16 code units (Lucene's default).
  std::size_t max_token_length = 255;

  // kPlain tokenizer + lowercase; no stopwords, no stemming.
  static AnalyzerConfig plain();
  // Anserini's default English analysis for BEIR "flat" indexes: UAX #29 tokenizer, possessive
  // filter, lowercase, Lucene English stopwords, Porter stemmer.
  static AnalyzerConfig anserini_english();
};

// Applies an AnalyzerConfig to text.
// Thread safety: immutable after construction; analyze() is safe to call concurrently.
class Analyzer {
 public:
  explicit Analyzer(AnalyzerConfig config = AnalyzerConfig::anserini_english());

  [[nodiscard]] std::vector<std::string> analyze(std::string_view text) const;
  [[nodiscard]] const AnalyzerConfig& config() const noexcept { return config_; }

 private:
  AnalyzerConfig config_;
};

// Porter stem of one (already lowercased) word, with Lucene PorterStemmer semantics.
// Words of length <= 2 are returned unchanged.
[[nodiscard]] std::string porter_stem(std::string_view word);

// True if `word` is in Lucene's English stopword set.
[[nodiscard]] bool is_lucene_english_stopword(std::string_view word) noexcept;

}  // namespace strata
