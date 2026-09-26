// Porter stemmer, ported from Lucene's org.apache.lucene.analysis.en.PorterStemmer (Apache 2.0),
// which follows Martin Porter's reference implementation. Operates on UTF-16 code units, like
// Java's char[], so results match Lucene for non-ASCII input too. Only ASCII suffixes are
// rewritten.

#include <string>

#include "text_util.hpp"

namespace strata {

namespace {

class PorterStemmer {
 public:
  explicit PorterStemmer(std::u16string word) : b_(std::move(word)) {}

  std::u16string stem() {
    k_ = static_cast<int>(b_.size()) - 1;
    k0_ = 0;
    if (k_ > k0_ + 1) {
      step1();
      step2();
      step3();
      step4();
      step5();
      step6();
    }
    b_.resize(static_cast<std::size_t>(k_ + 1));
    return b_;
  }

 private:
  [[nodiscard]] char16_t at(int i) const { return b_[static_cast<std::size_t>(i)]; }

  // cons(i) is true <=> b[i] is a consonant.
  [[nodiscard]] bool cons(int i) const {
    switch (at(i)) {
      case u'a':
      case u'e':
      case u'i':
      case u'o':
      case u'u':
        return false;
      case u'y':
        return i == k0_ ? true : !cons(i - 1);
      default:
        return true;
    }
  }

  // Number of consonant sequences between k0 and j: <c>(vc)^m<v>.
  [[nodiscard]] int m() const {
    int n = 0;
    int i = k0_;
    while (true) {
      if (i > j_) {
        return n;
      }
      if (!cons(i)) {
        break;
      }
      ++i;
    }
    ++i;
    while (true) {
      while (true) {
        if (i > j_) {
          return n;
        }
        if (cons(i)) {
          break;
        }
        ++i;
      }
      ++i;
      ++n;
      while (true) {
        if (i > j_) {
          return n;
        }
        if (!cons(i)) {
          break;
        }
        ++i;
      }
      ++i;
    }
  }

  [[nodiscard]] bool vowelinstem() const {
    for (int i = k0_; i <= j_; ++i) {
      if (!cons(i)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool doublec(int j) const {
    if (j < k0_ + 1) {
      return false;
    }
    if (at(j) != at(j - 1)) {
      return false;
    }
    return cons(j);
  }

  // i-2, i-1, i is consonant-vowel-consonant and the last is not w, x, or y.
  [[nodiscard]] bool cvc(int i) const {
    if (i < k0_ + 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) {
      return false;
    }
    const char16_t ch = at(i);
    return !(ch == u'w' || ch == u'x' || ch == u'y');
  }

  bool ends(std::u16string_view s) {
    const int l = static_cast<int>(s.size());
    const int o = k_ - l + 1;
    if (o < k0_) {
      return false;
    }
    for (int i = 0; i < l; ++i) {
      if (at(o + i) != s[static_cast<std::size_t>(i)]) {
        return false;
      }
    }
    j_ = k_ - l;
    return true;
  }

  // Sets (j+1)..k to s, readjusting k.
  void setto(std::u16string_view s) {
    const int l = static_cast<int>(s.size());
    const int o = j_ + 1;
    if (static_cast<std::size_t>(o + l) > b_.size()) {
      b_.resize(static_cast<std::size_t>(o + l));
    }
    for (int i = 0; i < l; ++i) {
      b_[static_cast<std::size_t>(o + i)] = s[static_cast<std::size_t>(i)];
    }
    k_ = j_ + l;
  }

  void r(std::u16string_view s) {
    if (m() > 0) {
      setto(s);
    }
  }

  // Plurals and -ed / -ing.
  void step1() {
    if (at(k_) == u's') {
      if (ends(u"sses")) {
        k_ -= 2;
      } else if (ends(u"ies")) {
        setto(u"i");
      } else if (at(k_ - 1) != u's') {
        --k_;
      }
    }
    if (ends(u"eed")) {
      if (m() > 0) {
        --k_;
      }
    } else if ((ends(u"ed") || ends(u"ing")) && vowelinstem()) {
      k_ = j_;
      if (ends(u"at")) {
        setto(u"ate");
      } else if (ends(u"bl")) {
        setto(u"ble");
      } else if (ends(u"iz")) {
        setto(u"ize");
      } else if (doublec(k_)) {
        const char16_t ch = at(k_--);
        if (ch == u'l' || ch == u's' || ch == u'z') {
          ++k_;
        }
      } else if (m() == 1 && cvc(k_)) {
        setto(u"e");
      }
    }
  }

  // Terminal y -> i when there is another vowel in the stem.
  void step2() {
    if (ends(u"y") && vowelinstem()) {
      b_[static_cast<std::size_t>(k_)] = u'i';
    }
  }

  // Double suffixes -> single ones.
  void step3() {
    if (k_ == k0_) {
      return;
    }
    switch (at(k_ - 1)) {
      case u'a':
        if (ends(u"ational")) {
          r(u"ate");
        } else if (ends(u"tional")) {
          r(u"tion");
        }
        break;
      case u'c':
        if (ends(u"enci")) {
          r(u"ence");
        } else if (ends(u"anci")) {
          r(u"ance");
        }
        break;
      case u'e':
        if (ends(u"izer")) {
          r(u"ize");
        }
        break;
      case u'l':
        if (ends(u"bli")) {
          r(u"ble");
        } else if (ends(u"alli")) {
          r(u"al");
        } else if (ends(u"entli")) {
          r(u"ent");
        } else if (ends(u"eli")) {
          r(u"e");
        } else if (ends(u"ousli")) {
          r(u"ous");
        }
        break;
      case u'o':
        if (ends(u"ization")) {
          r(u"ize");
        } else if (ends(u"ation")) {
          r(u"ate");
        } else if (ends(u"ator")) {
          r(u"ate");
        }
        break;
      case u's':
        if (ends(u"alism")) {
          r(u"al");
        } else if (ends(u"iveness")) {
          r(u"ive");
        } else if (ends(u"fulness")) {
          r(u"ful");
        } else if (ends(u"ousness")) {
          r(u"ous");
        }
        break;
      case u't':
        if (ends(u"aliti")) {
          r(u"al");
        } else if (ends(u"iviti")) {
          r(u"ive");
        } else if (ends(u"biliti")) {
          r(u"ble");
        }
        break;
      case u'g':
        if (ends(u"logi")) {
          r(u"log");
        }
        break;
      default:
        break;
    }
  }

  // -ic-, -full, -ness etc.
  void step4() {
    switch (at(k_)) {
      case u'e':
        if (ends(u"icate")) {
          r(u"ic");
        } else if (ends(u"ative")) {
          r(u"");
        } else if (ends(u"alize")) {
          r(u"al");
        }
        break;
      case u'i':
        if (ends(u"iciti")) {
          r(u"ic");
        }
        break;
      case u'l':
        if (ends(u"ical")) {
          r(u"ic");
        } else if (ends(u"ful")) {
          r(u"");
        }
        break;
      case u's':
        if (ends(u"ness")) {
          r(u"");
        }
        break;
      default:
        break;
    }
  }

  // -ant, -ence etc. in context <c>vcvc<v>.
  void step5() {
    if (k_ == k0_) {
      return;
    }
    bool matched = false;
    switch (at(k_ - 1)) {
      case u'a':
        matched = ends(u"al");
        break;
      case u'c':
        matched = ends(u"ance") || ends(u"ence");
        break;
      case u'e':
        matched = ends(u"er");
        break;
      case u'i':
        matched = ends(u"ic");
        break;
      case u'l':
        matched = ends(u"able") || ends(u"ible");
        break;
      case u'n':
        matched = ends(u"ant") || ends(u"ement") || ends(u"ment") || ends(u"ent");
        break;
      case u'o':
        matched = (ends(u"ion") && j_ >= 0 && (at(j_) == u's' || at(j_) == u't')) || ends(u"ou");
        break;
      case u's':
        matched = ends(u"ism");
        break;
      case u't':
        matched = ends(u"ate") || ends(u"iti");
        break;
      case u'u':
        matched = ends(u"ous");
        break;
      case u'v':
        matched = ends(u"ive");
        break;
      case u'z':
        matched = ends(u"ize");
        break;
      default:
        break;
    }
    if (matched && m() > 1) {
      k_ = j_;
    }
  }

  // Final -e and -ll.
  void step6() {
    j_ = k_;
    if (at(k_) == u'e') {
      const int a = m();
      if (a > 1 || (a == 1 && !cvc(k_ - 1))) {
        --k_;
      }
    }
    if (at(k_) == u'l' && doublec(k_) && m() > 1) {
      --k_;
    }
  }

  std::u16string b_;
  int k_ = 0;
  int k0_ = 0;
  int j_ = 0;
};

}  // namespace

std::string porter_stem(std::string_view word) {
  return text::to_utf8(PorterStemmer(text::to_utf16(word)).stem());
}

}  // namespace strata
