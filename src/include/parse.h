#pragma once

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

#include "ast.h"

namespace fs = std::filesystem;

struct Tok {

  // Available token types
  enum TokType {
    INVALID,
    DOT,
    ARITH_PLUS,
    LPAREN,
    RPAREN,
    BOOL_TRUE,
    BOOL_FALSE,
    ID,
    NUM,
    STRING,
    BYTE_STRING,
    SYMBOLMARK,
    LINKLET,
    DEFINE_VALUES,
    LAMBDA,
    LETREC_VALUES,
    BEGIN,
    BEGIN0,
    CASE_LAMBDA,
    IF,
    VOID,
    RAISE_ARGUMENT_ERROR,
    PROCEDURE_ARITY_INCLUDES_C,
    MAKE_STRUCT_TYPE,
    LET_VALUES,
    WITH_CONTINUATION_MARK,
    K_VARIABLE_REFERENCE,
    CHAR,
    CHAR_HEX,
    CHAR_NAMED,
    QUOTE,
    BYTE_REGEXP_LITERAL,
    REGEXP_LITERAL,
    HASH_START,
    HASHEQ_START,
    HASHEQV_START,
    KEYWORD,
    VECTOR_START,
    VALUES,
    SETBANG,
  };
  TokType tok;
  std::wstring value;
  size_t start; /// start position of token in stream
  size_t end;   /// end position of token in stream

  Tok() : tok(INVALID), value(), start(0), end(0) {}
  Tok(TokType tok, size_t pos) : tok(tok), value(), start(pos), end(pos) {}
  Tok(TokType tok, size_t start, size_t end)
      : tok(tok), value(), start(start), end(end) {}
  Tok(TokType tok, std::wstring_view val, size_t start, size_t end)
      : tok(tok), value(val), start(start), end(end) {}

  friend std::ostream &operator<<(std::ostream &, const Tok &);
  [[nodiscard]] bool isValid() const { return tok != TokType::INVALID; }
};

class ParseWindow {
public:
  ParseWindow(const std::wstring &Str);

  [[nodiscard]] bool isEmpty() const { return View.size() == 0; }
  [[nodiscard]] wchar_t peekChar() const {
    assert(!isEmpty());
    wchar_t c = View[0];
    return c;
  }
  [[nodiscard]] const std::wstring_view getSubview(size_t n) const {
    if (n <= size()) {
      return std::wstring_view(Raw.cbegin() + Pos, Raw.cbegin() + Pos + n);
    }
    return {};
  }
  bool isPrefix(const wchar_t *Str) const { return View.starts_with(Str); }
  wchar_t operator[](size_t N) const { return View[N]; }
  void skipPrefix(size_t N) {
    assert(View.size() >= N);
    Pos += N;
    View.remove_prefix(N);
  }
  [[nodiscard]] size_t size() const { return View.size(); }
  [[nodiscard]] size_t getPosition() const { return Pos; }
  void rewind(size_t N) {
    assert(N <= Pos);
    Pos -= N;
    View = std::wstring_view(Raw.cbegin() + Pos, Raw.cend());
  }
  void rewindTo(size_t P) {
    Pos = P;
    View = std::wstring_view(Raw.cbegin() + Pos, Raw.cend());
  }

  bool searchRegex(const wchar_t *Regex, std::wcmatch &CM) const;

private:
  const std::wstring &Raw; /// Raw initial string needed for rewind.
  std::wstring_view View;  /// View into the file that starts at pos and ends at
                           /// the end of the file.
  size_t Pos; /// Position in the file in characters that the windows starts at.
};

class Stream {
public:
  static std::wstring readFile(const fs::path &Path);

  explicit Stream(const fs::path &Path);
  explicit Stream(const wchar_t *View);
  explicit Stream(const char *View);
  [[nodiscard]] wchar_t peekChar(size_t n = 0) const;
  void skipPrefix(size_t N) { Win.skipPrefix(N); }
  [[nodiscard]] std::wstring_view getSubview(size_t N) const {
    return Win.getSubview(N);
  }
  std::wstring_view getSubviewAndSkip(size_t N) {
    std::wstring_view View = getSubview(N);
    skipPrefix(N);
    return View;
  }
  void rewind(const Tok &T);
  void rewind(size_t Len);
  void rewindTo(size_t Pos);

  void skipWhitespace();
  const wchar_t &operator[](size_t Idx) const { return Contents[Idx]; }
  bool searchRegex(const wchar_t *Regex, std::wcmatch &CM) const;
  bool searchString(const wchar_t *Str) const;
  [[nodiscard]] size_t getPosition() const;
  [[nodiscard]] bool isEmpty() const { return Win.isEmpty(); }

private:
  fs::path Path;
  std::wstring Contents;
  ParseWindow Win; /// Window into the string currently seen by the parser.
};

Tok gettok(Stream &S);
std::optional<Tok> maybeLexIdOrNumber(Stream &S);
std::optional<Tok> maybeLexSymbolMark(Stream &S);
std::optional<Tok> maybeLexString(Stream &S);
std::optional<Tok> maybeLexSchemeChar(Stream &S);
std::optional<Tok> maybeLexBoolean(Stream &S);
std::optional<Tok> maybeLexHash(Stream &S);
std::optional<Tok> maybeLexKeyword(Stream &S);
std::optional<Tok> maybeLexRegexpLiteral(Stream &S);
std::optional<Tok> maybeLexVector(Stream &S);
bool isExplicitSign(Stream &S, size_t Offset = 0);
bool isSpecialSubsequent(Stream &S, size_t Offset = 0);
bool isSubsequent(Stream &S, size_t Offset = 0);
bool isDigit(Stream &S, size_t Offset = 0);
bool isInitial(Stream &S, size_t Offset = 0);
bool isSignSubsequent(Stream &S, size_t Offset = 0);
bool isSymbolElement(Stream &S, size_t Offset = 0);
bool isHexDigit(Stream &S, size_t Offset = 0);
size_t isHexScalarValue(Stream &S, size_t Offset = 0);
bool isMnemonicEscape(Stream &S, size_t Offset = 0);
size_t isInlineHexEscape(Stream &S, size_t Offset = 0);
bool isLetter(Stream &S, size_t Offset = 0);
bool isSpecialInitial(Stream &S, size_t Offset = 0);
bool isDotSubsequent(Stream &S, size_t Offset = 0);
size_t isPeculiarIdentifier(Stream &S, size_t Offset = 0);
bool isDelimiter(Stream &S, size_t Offset = 0);

// Parsing functions:
std::unique_ptr<ast::Lambda> parseLambda(Stream &S);
std::unique_ptr<ast::Linklet> parseLinklet(Stream &S);
std::unique_ptr<ast::Begin> parseBegin(Stream &S);