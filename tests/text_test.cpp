#include "strata/text.hpp"

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

namespace strata {
namespace {

using Tokens = std::vector<std::string>;

Tokens words(std::string_view text) {
  return Analyzer({.tokenizer = Tokenizer::kUnicodeWords}).analyze(text);
}

// --- UAX #29 tokenizer (Lucene StandardTokenizer behaviour) ----------

TEST(UnicodeWords, PunctuationAndWhitespaceSeparate) {
  EXPECT_EQ(words("Hello, World!  foo\tbar\n"), (Tokens{"hello", "world", "foo", "bar"}));
  EXPECT_EQ(words("...!?"), Tokens{});
  EXPECT_EQ(words(""), Tokens{});
}

TEST(UnicodeWords, MidWordPunctuationJoinsLetters) {
  EXPECT_EQ(words("don't rock'n'roll U.S.A. x.com a:b"),
            (Tokens{"don't", "rock'n'roll", "u.s.a", "x.com", "a:b"}));
  // Trailing / leading punctuation is not part of the word.
  EXPECT_EQ(words("'quoted' end. cats'"), (Tokens{"quoted", "end", "cats"}));
}

TEST(UnicodeWords, MidNumPunctuationJoinsDigits) {
  EXPECT_EQ(words("3.14 1,000,000 p<0.05 1;2"), (Tokens{"3.14", "1,000,000", "p", "0.05", "1;2"}));
  EXPECT_EQ(words("a,b"), (Tokens{"a", "b"}));  // ',' joins digits only
  EXPECT_EQ(words("1:2"), (Tokens{"1", "2"}));  // ':' joins letters only
}

TEST(UnicodeWords, LettersDigitsAndUnderscore) {
  EXPECT_EQ(words("covid-19 IL-6R β2 foo_bar __init__"),
            (Tokens{"covid", "19", "il", "6r", "β2", "foo_bar", "__init__"}));
  EXPECT_EQ(words("___"), Tokens{});  // no letter or digit: not a token
}

TEST(UnicodeWords, NonLatinScripts) {
  EXPECT_EQ(words("αβγ-δ Москва café"), (Tokens{"αβγ", "δ", "москва", "café"}));
  EXPECT_EQ(words("中文テキスト"), (Tokens{"中", "文", "テキスト"}));  // Han per char, Katakana run
  EXPECT_EQ(words("ひらがな"), (Tokens{"ひ", "ら", "が", "な"}));
}

TEST(UnicodeWords, EmojiAreTheirOwnTokens) {
  EXPECT_EQ(words("Tween® 80 © 2020 A↔B"),
            (Tokens{"tween", "®", "80", "©", "2020", "a", "↔", "b"}));
  // Skin-tone modifier, ZWJ sequence, and a flag (two regional indicators) stay whole.
  EXPECT_EQ(words("👍🏽 ok 👨‍💻 🇺🇸"), (Tokens{"👍🏽", "ok", "👨‍💻", "🇺🇸"}));
  EXPECT_EQ(words("3 # *"), (Tokens{"3"}));  // ASCII Emoji=Yes characters are not emoji tokens
}

TEST(UnicodeWords, CombiningMarksStayAttached) {
  EXPECT_EQ(words("café x"), (Tokens{"café", "x"}));
}

TEST(UnicodeWords, InvalidUtf8IsASeparator) {
  EXPECT_EQ(words("ab\xff\xfe"
                  "cd"),
            (Tokens{"ab", "cd"}));
}

TEST(UnicodeWords, OverlongTokensAreSplit) {
  AnalyzerConfig config{.tokenizer = Tokenizer::kUnicodeWords, .max_token_length = 4};
  EXPECT_EQ(Analyzer(config).analyze("abcdefghij"), (Tokens{"abcd", "efgh", "ij"}));
}

TEST(PlainTokenizer, SplitsOnEverythingButLettersAndDigits) {
  const Analyzer plain(AnalyzerConfig::plain());
  EXPECT_EQ(plain.analyze("don't 3.14 foo_bar U.S. Café"),
            (Tokens{"don", "t", "3", "14", "foo", "bar", "u", "s", "café"}));
}

// Arbitrary bytes (invalid UTF-8, truncated sequences, NULs) must never crash or read out of
// bounds (run under the asan preset), and output tokens must be valid, non-empty UTF-8.
TEST(Analyzer, SurvivesRandomBytes) {
  std::mt19937 rng(99);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> length(0, 64);
  const Analyzer anserini(AnalyzerConfig::anserini_english());
  const Analyzer plain(AnalyzerConfig::plain());
  for (int trial = 0; trial < 2000; ++trial) {
    std::string input(static_cast<std::size_t>(length(rng)), '\0');
    for (char& c : input) {
      c = static_cast<char>(byte(rng));
    }
    for (const Analyzer* a : {&anserini, &plain}) {
      for (const auto& token : a->analyze(input)) {
        ASSERT_FALSE(token.empty());
      }
    }
  }
}

// --- Filters ----------

TEST(Analyzer, PossessiveFilterRunsBeforeLowercase) {
  const Analyzer a({.tokenizer = Tokenizer::kUnicodeWords, .english_possessive = true});
  EXPECT_EQ(a.analyze("John's JOHN'S john’s it's"), (Tokens{"john", "john", "john", "it"}));
}

TEST(Analyzer, LowercaseIsOptional) {
  const Analyzer a({.tokenizer = Tokenizer::kUnicodeWords, .lowercase = false});
  EXPECT_EQ(a.analyze("Hello WORLD"), (Tokens{"Hello", "WORLD"}));
}

TEST(Analyzer, LuceneEnglishStopwords) {
  EXPECT_TRUE(is_lucene_english_stopword("the"));
  EXPECT_TRUE(is_lucene_english_stopword("with"));
  EXPECT_FALSE(is_lucene_english_stopword("was not"));
  EXPECT_FALSE(is_lucene_english_stopword("has"));  // not in Lucene's 33-word list
  const Analyzer a({.tokenizer = Tokenizer::kUnicodeWords, .stopwords = Stopwords::kLuceneEnglish});
  EXPECT_EQ(a.analyze("The cat is on THE mat"), (Tokens{"cat", "mat"}));
}

TEST(Analyzer, AnseriniEnglishPipeline) {
  const Analyzer a(AnalyzerConfig::anserini_english());
  EXPECT_EQ(a.analyze("The cats' toys aren't running; John's generalizations of oscillators"),
            (Tokens{"cat", "toi", "aren't", "run", "john", "gener", "oscil"}));
}

// --- Porter ----------

// Expected stems generated with NLTK's PorterStemmer(mode=MARTIN_EXTENSIONS), an independent
// implementation of the same (reference) algorithm Lucene ports. Covers every rule of steps 1-6.
TEST(Porter, MatchesReferenceAlgorithm) {
  const std::vector<std::pair<std::string, std::string>> cases = {{"caresses", "caress"},
                                                                  {"ponies", "poni"},
                                                                  {"ties", "ti"},
                                                                  {"caress", "caress"},
                                                                  {"cats", "cat"},
                                                                  {"feed", "feed"},
                                                                  {"agreed", "agre"},
                                                                  {"disabled", "disabl"},
                                                                  {"matting", "mat"},
                                                                  {"mating", "mate"},
                                                                  {"meeting", "meet"},
                                                                  {"milling", "mill"},
                                                                  {"messing", "mess"},
                                                                  {"meetings", "meet"},
                                                                  {"happy", "happi"},
                                                                  {"sky", "sky"},
                                                                  {"relational", "relat"},
                                                                  {"conditional", "condit"},
                                                                  {"rational", "ration"},
                                                                  {"valenci", "valenc"},
                                                                  {"hesitanci", "hesit"},
                                                                  {"digitizer", "digit"},
                                                                  {"conformabli", "conform"},
                                                                  {"radicalli", "radic"},
                                                                  {"differentli", "differ"},
                                                                  {"vileli", "vile"},
                                                                  {"analogousli", "analog"},
                                                                  {"vietnamization", "vietnam"},
                                                                  {"predication", "predic"},
                                                                  {"operator", "oper"},
                                                                  {"feudalism", "feudal"},
                                                                  {"decisiveness", "decis"},
                                                                  {"hopefulness", "hope"},
                                                                  {"callousness", "callous"},
                                                                  {"formaliti", "formal"},
                                                                  {"sensitiviti", "sensit"},
                                                                  {"sensibiliti", "sensibl"},
                                                                  {"triplicate", "triplic"},
                                                                  {"formative", "form"},
                                                                  {"formalize", "formal"},
                                                                  {"electriciti", "electr"},
                                                                  {"electrical", "electr"},
                                                                  {"hopeful", "hope"},
                                                                  {"goodness", "good"},
                                                                  {"revival", "reviv"},
                                                                  {"allowance", "allow"},
                                                                  {"inference", "infer"},
                                                                  {"airliner", "airlin"},
                                                                  {"gyroscopic", "gyroscop"},
                                                                  {"adjustable", "adjust"},
                                                                  {"defensible", "defens"},
                                                                  {"irritant", "irrit"},
                                                                  {"replacement", "replac"},
                                                                  {"adjustment", "adjust"},
                                                                  {"dependent", "depend"},
                                                                  {"adoption", "adopt"},
                                                                  {"homologou", "homolog"},
                                                                  {"communism", "commun"},
                                                                  {"activate", "activ"},
                                                                  {"angulariti", "angular"},
                                                                  {"homologous", "homolog"},
                                                                  {"effective", "effect"},
                                                                  {"bowdlerize", "bowdler"},
                                                                  {"probate", "probat"},
                                                                  {"rate", "rate"},
                                                                  {"cease", "ceas"},
                                                                  {"controll", "control"},
                                                                  {"roll", "roll"},
                                                                  {"generalizations", "gener"},
                                                                  {"oscillators", "oscil"},
                                                                  {"analogies", "analog"},
                                                                  {"probably", "probabl"},
                                                                  {"sensibility", "sensibl"},
                                                                  {"is", "is"},
                                                                  {"as", "as"},
                                                                  {"a", "a"}};
  for (const auto& [word, stem] : cases) {
    EXPECT_EQ(porter_stem(word), stem) << word;
  }
}

TEST(Porter, NonAsciiAndEdgeCases) {
  EXPECT_EQ(porter_stem(""), "");
  EXPECT_EQ(porter_stem("ab"), "ab");           // length <= 2 unchanged
  EXPECT_EQ(porter_stem("cafés"), "café");      // plural s removed, é is a consonant
  EXPECT_EQ(porter_stem("naïveness"), "naïv");  // suffix rules still apply
  EXPECT_EQ(porter_stem("москва"), "москва");   // no ASCII suffix: unchanged
}

}  // namespace
}  // namespace strata
