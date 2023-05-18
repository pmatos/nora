#include "parse.h"

#include <array>
#include <cassert>
#include <codecvt>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "Casting.h"
#include "idpool.h"

// This is the main parse file that parses linklets generated by Racket
// Racket docs do not have a formal grammar in one place and it's indeed split
// into different pages. I have tried to collect all the relevant information
// here.
//
// Linklet grammar starts in
// https://docs.racket-lang.org/reference/linklets.html#%28tech._linklet%29
// but then you're forward to
// https://docs.racket-lang.org/reference/syntax-model.html#%28part._fully-expanded%29
// and this one still misses some important information, like the definition of
// datum for example.
//
// The following text will try to summarize what I understand from the docs, and
// as a reference for what I have implemented.
//
// Linklet grammar:
//
// linklet := (linklet [[<imported-id/renamed-id> ...] ...]
//                     [<exported-id/renamed> ...]
//               <defn-or-expr> ...)
//
// imported-id/renamed := <imported-id>
//                      | (<external-imported-id> <internal-imported-id>)
//
// exported-id/renamed := <exported-id>
//                      | (<internal-exported-id> <external-exported-id>)
//
// defn-or-expr := <defn> | <expr>                          - parseDefnOrExpr
//
// defn := (define-values (<id> ...) <expr>)                - parseDefineValues
//       | (define-syntaxes (<id> ...) <expr>)
//
// expr := <id>                                             - parseIdentifier
//       | (lambda <formals> <expr>)                        - parseLambda
//       | (case-lambda (<formals> <expr>) ...)
//       | (if <expr> <expr> <expr>)                        - parseIfCond
//       | (begin <expr> ...+)                              - parseBegin
//       | (begin0 <expr> ...+)                             - parseBegin
//       | (let-values ([<id> ...) <expr>] ...) <expr>)     - parseLetValues
//       | (letrec-values ([(<id> ...) <expr>] ...) <expr>)
//       | (set! <id> <expr>)                               - parseSetBang
//       | (quote <datum>)
//       | (with-continuation-mark <expr> <expr> <expr>)
//       | (<expr> ...+)                                    - parseApplication
//       | (%variable-reference <id>)
//       | (%variable-reference (%top . id))  <- Allowed?
//       | (%variable-reference)              <- Allowed?
//
// formals := <id>                                          - parseFormals
//          | (<id> ...+ . id)
//          | (id ...)
//
// <id> := TODO
//
// <datum> := <self-quoting-datum> | <character> | TODO
//
// <character> := TODO
//
// <self-quoting-datum> := <boolean> | <number> | <string> | <byte-string>
//
// <boolean> := #t | #f
//
// <number> := <integer> | TODO
//
// <string> := TODO
//
// <byte-string> := TODO
//
// The following is a list of parsed runtime library functions that are not part
// of the core grammar:
// - values

ParseWindow::ParseWindow(const std::wstring &Str)
    : Raw(Str), View(Str), Pos(0) {}

Stream::Stream(const fs::path &Path)
    : Path(Path), Contents(Stream::readFile(Path)), Win(Contents) {}
Stream::Stream(const wchar_t *View) : Path(), Contents(View), Win(Contents) {}

/// Read the file pointed to by \param path and returns its contents as a
/// string.
std::wstring Stream::readFile(const fs::path &Path) {
  // Open the stream to 'lock' the file.
  std::wifstream F(Path, std::ios::in);
  if (!F.good()) {
    std::cerr << "cannot open " << Path << std::endl;
    exit(EXIT_FAILURE);
  }

  // Racket source files are UTF-8 encoded.
  //  std::codecvt_utf8<wchar_t> Facet;
  //  std::locale L(F.getloc(), &Facet);
  //  F.imbue(L);

  // Obtain the size of the file.
  const std::uintmax_t Sz = fs::file_size(Path);

  // Create a buffer.
  std::wstring Result(Sz, '\0');

  // Read the whole file into the buffer.
  F.read(Result.data(), static_cast<std::streamsize>(Sz));

  return Result;
}

std::ostream &operator<<(std::ostream &Out, const Tok &T) { return Out; }

size_t Stream::getPosition() const { return Win.getPosition(); }

bool Stream::searchRegex(const wchar_t *Regex, std::wcmatch &CM) const {
  return Win.searchRegex(Regex, CM);
}

bool ParseWindow::searchRegex(const wchar_t *Regex, std::wcmatch &CM) const {
  std::wregex E(Regex);

  return std::regex_search(View.cbegin(), View.cend(), CM, E,
                           std::regex_constants::match_continuous);
}

bool Stream::searchString(const wchar_t *Str) const {
  return Win.isPrefix(Str);
}

wchar_t Stream::peekChar(size_t N) const {
  if (N < Win.size()) {
    return Win[N];
  }
  return EOF;
}

/// Skips whitespace in stream
void Stream::skipWhitespace() {
  if (isEmpty()) {
    return;
  }

  // Lets eat comments now
  if (Win.peekChar() == ';') {
    while (Win.peekChar() != '\n' && !Win.isEmpty()) {
      Win.skipPrefix(1);
    }
  }

  // Lets eat spaces - including the previously left \n at the end of the line
  // comment
  while (isspace(Win[0])) {
    Win.skipPrefix(1);
  }

  if (Win[0] == ';') {
    skipWhitespace();
  }
}

void Stream::rewind(const Tok &T) {
  ssize_t Tlen = T.end - T.start + 1;
  Win.rewind(Tlen);
}

void Stream::rewind(size_t Len) { Win.rewind(Len); }
void Stream::rewindTo(size_t Pos) { Win.rewindTo(Pos); }

// Linklet parsing code

// Lexer part where we split the input stream into tokens

// From:
// https://docs.racket-lang.org/reference/reader.html#%28part._parse-symbol%29
// which says (12/2022):
// A sequence that does not start with a delimiter or # is parsed as either a
// symbol, a number, or a extflonum, except that . by itself is never parsed
// as a symbol or number (unless the read-accept-dot parameter is set to #f).
// A successful number or extflonum parse takes precedence over a symbol parse.
// A #% also starts a symbol. The resulting symbol is interned. See the start
// of Delimiters and Dispatch for information about | and \ in parsing symbols.
std::optional<Tok> maybeLexIdOrNumber(Stream &S) {
  if (isDelimiter(S) || (S.peekChar() == '#' && S.peekChar(1) != '%')) {
    return {};
  }

  size_t Start = S.getPosition();
  // <vertical line> <symbol element> <vertical line>
  if (S.peekChar() == '|') {
    size_t Count = 1;
    while (isSymbolElement(S, Count)) {
      Count++;
    }
    if (S.peekChar(Count++) != '|') {
      return {};
    }

    std::wstring_view Value = S.getSubviewAndSkip(Count);

    bool isNumber = std::regex_match(Value.cbegin(), Value.cend(),
                                     std::wregex(L"[-+]?[0-9]+"));
    return {Tok(isNumber ? Tok::TokType::NUM : Tok::TokType::ID, Value, Start,
                Start + Count - 1)};
  }

  // or
  // just parse until we find a delimiter
  size_t Count = 0;
  while (!isDelimiter(S, Count) && S.peekChar(Count) != EOF &&
         !iswspace(S.peekChar(Count))) {
    Count++;
  }

  // also cannot be a single dot
  if (Count == 0) {
    return {};
  }
  if (Count == 1 && S.peekChar() == '.') {
    return {};
  }

  std::wstring_view Value = S.getSubviewAndSkip(Count);
  bool isNumber = std::regex_match(Value.cbegin(), Value.cend(),
                                   std::wregex(L"[-+]?[0-9]+"));
  Tok T(isNumber ? Tok::TokType::NUM : Tok::TokType::ID, Value, Start,
        Start + Count - 1);

  // Transform token
  std::array KnownTokens = {
      std::make_pair(L"lambda", Tok::TokType::LAMBDA),
      std::make_pair(L"quote", Tok::TokType::QUOTE),
      std::make_pair(L"define-values", Tok::TokType::DEFINE_VALUES),
      std::make_pair(L"values", Tok::TokType::VALUES),
      std::make_pair(L"linklet", Tok::TokType::LINKLET),
      std::make_pair(L"letrec-values", Tok::TokType::LETREC_VALUES),
      std::make_pair(L"begin", Tok::TokType::BEGIN),
      std::make_pair(L"begin0", Tok::TokType::BEGIN0),
      std::make_pair(L"set!", Tok::TokType::SETBANG),
      std::make_pair(L"case-lambda", Tok::TokType::CASE_LAMBDA),
      std::make_pair(L"if", Tok::TokType::IF),
      std::make_pair(L"void", Tok::TokType::VOID),
      std::make_pair(L"raise-argument-error",
                     Tok::TokType::RAISE_ARGUMENT_ERROR),
      std::make_pair(L"procedure-arity-includes/c",
                     Tok::TokType::PROCEDURE_ARITY_INCLUDES_C),
      std::make_pair(L"make-struct-type", Tok::TokType::MAKE_STRUCT_TYPE),
      std::make_pair(L"let-values", Tok::TokType::LET_VALUES),
      std::make_pair(L"with-continuation-mark",
                     Tok::TokType::WITH_CONTINUATION_MARK),
      std::make_pair(L"#%variable-reference",
                     Tok::TokType::K_VARIABLE_REFERENCE)};

  if (T.tok == Tok::TokType::ID) {
    for (auto [Name, Token] : KnownTokens) {
      if (T.value == Name) {
        T.tok = Token;
        break;
      }
    }
  }

  return {T};
}

std::optional<Tok> maybeLexKeyword(Stream &S) {
  size_t Start = S.getPosition();
  if (!S.searchString(L"#:")) {
    return {};
  }
  S.skipPrefix(2);

  std::optional<Tok> Identifier = maybeLexIdOrNumber(S);
  if (!Identifier) {
    S.rewind(2);
    return {};
  }

  Tok T(Tok::TokType::KEYWORD, Identifier->value, Start, Identifier->end);
  return {T};
}

std::optional<Tok> maybeLexVector(Stream &S) {
  size_t Start = S.getPosition();
  if (!S.searchString(L"#(")) {
    return {};
  }
  S.skipPrefix(2);

  Tok T(Tok::TokType::VECTOR_START, Start, S.getPosition());
  return {T};
}

std::optional<Tok> maybeLexBoolean(Stream &S) {
  if (S.searchString(L"#t") || S.searchString(L"#f")) {
    Tok T(S.peekChar(1) == 't' ? Tok::TokType::BOOL_TRUE
                               : Tok::TokType::BOOL_FALSE,
          S.getSubviewAndSkip(2), S.getPosition(), S.getPosition() + 1);
    return {T};
  }

  return {};
}

std::optional<Tok> maybeLexHash(Stream &S) {

  if (S.searchString(L"#hasheqv(")) {
    Tok T(Tok::TokType::HASHEQV_START, S.getPosition(), S.getPosition() + 9);
    S.skipPrefix(9);
    return {T};
  }

  if (S.searchString(L"#hasheq(")) {
    Tok T(Tok::TokType::HASHEQ_START, S.getPosition(), S.getPosition() + 8);
    S.skipPrefix(8);
    return {T};
  }

  if (S.searchString(L"#hash(")) {
    Tok T(Tok::TokType::HASH_START, S.getPosition(), S.getPosition() + 6);
    S.skipPrefix(6);
    return {T};
  }

  return {};
}

std::optional<Tok> maybeLexDot(Stream &S) {
  if (S.searchString(L".")) {
    Tok T(Tok::TokType::DOT, L".", S.getPosition(), S.getPosition());
    S.skipPrefix(1);
    return {T};
  }
  return {};
}

std::optional<Tok> maybeLexString(Stream &S) {
  std::wcmatch CM;
  if (S.searchRegex(LR"|("([^"\\]*(\\.[^"\\]*)*)")|", CM)) {
    Tok T(Tok::TokType::STRING, CM.str(), S.getPosition(),
          S.getPosition() + CM.length() - 1);
    S.skipPrefix(CM.length());
    return {T};
  }

  return {};
}

std::optional<Tok> maybeLexSymbolMark(Stream &S) {
  if (S.peekChar() != '\'') {
    return {};
  }

  Tok T(Tok::TokType::SYMBOLMARK, S.getPosition());
  S.skipPrefix(1);
  return T;
}

std::optional<Tok> maybeLexRegexpLiteral(Stream &S) {
  // Parses a byte regexp literal in racket
  // #rx# followed a by "-delimited string containing the pattern.
  size_t Start = S.getPosition();
  if (!S.searchString(L"#rx")) {
    return {};
  }
  S.skipPrefix(3);

  bool IsByteRegexp = S.peekChar() == '#';
  if (IsByteRegexp) {
    S.skipPrefix(1);
  }

  std::optional<Tok> Str = maybeLexString(S);
  if (!Str) {
    if (IsByteRegexp) {
      S.rewind(4);
    } else {
      S.rewind(3);
    }
    return {};
  }

  Tok T(IsByteRegexp ? Tok::TokType::BYTE_REGEXP_LITERAL
                     : Tok::TokType::REGEXP_LITERAL,
        Str->value, Start, Str->end);
  return {T};
}

std::optional<Tok> maybeLexByteString(Stream &S) {
  // Parses a byte string literal in racket
  // # followed a by "-delimited string containing the bytes.
  size_t Start = S.getPosition();
  if (S.peekChar() != '#') {
    return {};
  }
  S.skipPrefix(1);

  std::optional<Tok> Str = maybeLexString(S);
  if (!Str) {
    S.rewind(1);
    return {};
  }

  Tok T(Tok::TokType::BYTE_STRING, Str->value, Start, Str->end);
  return {T};
}

bool isCharacterName(Stream &S, std::wstring &Found) {
  static const std::array Names = {L"space",     L"newline", L"alarm",
                                   L"backspace", L"delete",  L"escape",
                                   L"null",      L"return",  L"tab"};

  for (const auto *Name : Names) {
    if (S.searchString(Name)) {
      Found = Name;
      return true;
    }
  }
  return false;
}

std::optional<Tok> maybeLexSchemeChar(Stream &S) {
  // A character looks like this
  // #\ followed by the represented char
  // #\u followed by the scalar value as hex number
  // #\<character name>
  if (S.getSubview(2) != L"#\\") {
    return {};
  }
  S.skipPrefix(2);

  std::wstring MaybeName;
  if (isCharacterName(S, MaybeName)) {
    std::wstring_view Value = S.getSubviewAndSkip(MaybeName.size());
    Tok T(Tok::TokType::CHAR_NAMED, Value, S.getPosition() - 2,
          S.getPosition());
    return {T};
  }

  std::wcmatch CM;
  if (S.searchRegex(LR"((?:u|x|U|X)[0-9a-fA-F]{4})", CM)) {
    Tok T(Tok::TokType::CHAR_HEX, CM.str(), S.getPosition() - 2,
          S.getPosition() + CM.length() - 1);
    S.skipPrefix(CM.length());
    return {T};
  }

  // A character is a literal wide character or the unicode replacement
  // character
  if ((std::iswgraph(S.peekChar()) || S.peekChar() == L'\uFFFD') &&
      (isDelimiter(S, 1) || S.peekChar(1) == EOF ||
       std::iswspace(S.peekChar(1)))) {
    Tok T(Tok::TokType::CHAR, S.getSubview(1), S.getPosition() - 2,
          S.getPosition());
    S.skipPrefix(1);
    return {T};
  }

  return {};
}

bool isLetter(Stream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
  case 'g':
  case 'h':
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
  case 'o':
  case 'p':
  case 'q':
  case 'r':
  case 's':
  case 't':
  case 'u':
  case 'v':
  case 'w':
  case 'x':
  case 'y':
  case 'z':
  case 'A':
  case 'B':
  case 'C':
  case 'D':
  case 'E':
  case 'F':
  case 'G':
  case 'H':
  case 'I':
  case 'J':
  case 'K':
  case 'L':
  case 'M':
  case 'N':
  case 'O':
  case 'P':
  case 'Q':
  case 'R':
  case 'S':
  case 'T':
  case 'U':
  case 'V':
  case 'W':
  case 'X':
  case 'Y':
  case 'Z':
    return true;
  default:
    return false;
  }
  return false;
}

bool isSpecialInitial(Stream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case '!':
  case '$':
  case '%':
  case '&':
  case '*':
  case '/':
  case ':':
  case '<':
  case '=':
  case '>':
  case '?':
  case '^':
  case '_':
  case '~':
    return true;
  default:
    return false;
  }
}

bool isInitial(Stream &S, size_t Offset) {
  if (isLetter(S, Offset) || isSpecialInitial(S, Offset)) {
    return true;
  }
  return false;
}

bool isDigit(Stream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    return true;
  default:
    return false;
  }
}

bool isExplicitSign(Stream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case '+':
  case '-':
    return true;
  default:
    return false;
  }
}

bool isSymbolElement(Stream &S, size_t Offset) {
  // a symbol element is either:
  // <inline hex escape> | <mnemonic escape> | \|
  if (isInlineHexEscape(S, Offset) || isMnemonicEscape(S, Offset) ||
      (S.peekChar(Offset) == '\\' && S.peekChar(Offset + 1) == '|')) {
    return true;
  }

  // or:
  // any other character other than |
  if (S.peekChar(Offset) != '|') {
    return true;
  }
  return false;
}

bool isHexDigit(Stream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
  case 'A':
  case 'B':
  case 'C':
  case 'D':
  case 'E':
  case 'F':
    return true;
  default:
    return false;
  }
}

size_t isHexScalarValue(Stream &S, size_t Offset) {
  size_t Count = 0;
  while (isHexDigit(S, Offset + Count)) {
    Count++;
  }
  return Count;
}

bool isMnemonicEscape(Stream &s, size_t offset) {
  if (s.peekChar(offset) == '\\' &&
      (s.peekChar(offset + 1) == 'a' || s.peekChar(offset + 1) == 'b' ||
       s.peekChar(offset + 1) == 't' || s.peekChar(offset + 1) == 'n' ||
       s.peekChar(offset + 1) == 'r')) {
    return true;
  }
  return false;
}

size_t isInlineHexEscape(Stream &S, size_t Offset) {
  if (S.peekChar(Offset) == '\\' && S.peekChar(Offset + 1) == 'x') {
    size_t Count = 2;
    if (size_t HexCount = isHexScalarValue(S, Offset + Count)) {
      Count += HexCount;
    } else {
      return 0;
    }
    if (S.peekChar(Offset + Count) == ';') {
      return Count + 1;
    }
  }
  return 0;
}

bool isDelimiter(Stream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case '(':
  case ')':
  case '[':
  case ']':
  case '{':
  case '}':
  case '"':
  case ',':
  case '\'':
  case '`':
  case ';':
    return true;
  default:
    return false;
  }
}

// The tokenization of a Racket file is based on the Reader
// reference documentation:
// https://docs.racket-lang.org/reference/reader.html
//
// From a high level view, the tokenization is done by
// looking at the first few characters.
// Symbols are sort of the catch all situation where if
// we cannot tokenize as anything else, then it's most likely
// a symbol unless it's a delimiter or whitespace, or etc, etc.
Tok gettok(Stream &S) {
  // We start by skipping whitespace which is not part of a token
  S.skipWhitespace();

  switch (S.peekChar()) {
  case '(':
    S.skipPrefix(1);
    return {Tok::TokType::LPAREN, S.getPosition()};
  case ')':
    S.skipPrefix(1);
    return {Tok::TokType::RPAREN, S.getPosition()};
  }

  std::optional<Tok> MaybeTok = maybeLexSchemeChar(S);
  if (!MaybeTok) {
    MaybeTok = maybeLexSymbolMark(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexByteString(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexString(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexRegexpLiteral(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexDot(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexBoolean(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexHash(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexKeyword(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexVector(S);
  }
  if (!MaybeTok) {
    MaybeTok = maybeLexIdOrNumber(S);
  }

  Tok T = MaybeTok.value_or(Tok());

  return T;
}

// Parsing functions
// Forward declarations
std::unique_ptr<ast::TLNode> parseDefn(Stream &S);
std::unique_ptr<ast::ExprNode> parseExpr(Stream &S);
std::unique_ptr<ast::Identifier> parseIdentifier(Stream &S);
std::unique_ptr<ast::Values> parseValues(Stream &S);
std::unique_ptr<ast::DefineValues> parseDefineValues(Stream &S);
std::unique_ptr<ast::Formal> parseFormals(Stream &S);
std::unique_ptr<ast::Begin> parseBegin(Stream &S);
std::unique_ptr<ast::Application> parseApplication(Stream &S);
std::unique_ptr<ast::SetBang> parseSetBang(Stream &S);
std::unique_ptr<ast::IfCond> parseIfCond(Stream &S);
std::unique_ptr<ast::BooleanLiteral> parseBooleanLiteral(Stream &S);
std::unique_ptr<ast::LetValues> parseLetValues(Stream &S);

std::vector<ast::Linklet::idpair_t> parseLinkletExports(Stream &S) {
  // parse a sequence of
  // (internal-exported-id external-exported-id)
  std::vector<ast::Linklet::idpair_t> Vec;

  while (true) {
    Tok T = gettok(S);
    if (T.tok != Tok::TokType::LPAREN) {
      S.rewind(T);
      return Vec;
    }

    // parse internal-exported-id
    std::optional<Tok> T1 = maybeLexIdOrNumber(S);
    if (!T1) {
      return {};
    }

    // parse external-exported-id
    std::optional<Tok> T2 = maybeLexIdOrNumber(S);
    if (!T2) {
      return {};
    }

    T = gettok(S);
    if (T.tok != Tok::TokType::RPAREN) {
      return {};
    }

    IdPool &IP = IdPool::instance();

    Vec.push_back(std::make_pair(IP.create(T1.value().value),
                                 IP.create(T2.value().value)));
  }

  return Vec;
}

std::unique_ptr<ast::TLNode> parseDefn(Stream &S) {
  auto DefnValues = parseDefineValues(S);
  if (!DefnValues) {
    return nullptr;
  }

  return DefnValues;
}

std::unique_ptr<ast::TLNode> parseDefnOrExpr(Stream &S) {
  std::unique_ptr<ast::TLNode> D = parseDefn(S);
  if (D) {
    return D;
  }
  std::unique_ptr<ast::ExprNode> E = parseExpr(S);
  return dyn_castU<ast::TLNode>(E);
}

std::unique_ptr<ast::Integer> parseInteger(Stream &S) {
  std::optional<Tok> Num = gettok(S);
  if (Num->tok != Tok::TokType::NUM) {
    S.rewind(*Num);
    return nullptr;
  }

  // We know the wstring is mostly just digits with possibly a prefix minus
  // so convert to string.
  std::wstring_view V = Num.value().value;
  std::string NumStr(V.cbegin(), V.cend());

  return std::make_unique<ast::Integer>(NumStr);
}

// An expression is either:
// - an integer
// - an identifier
// - a values
// - an arithmetic plus
// - a lambda
// - a begin
std::unique_ptr<ast::ExprNode> parseExpr(Stream &S) {
  std::unique_ptr<ast::Integer> I = parseInteger(S);
  if (I) {
    return I;
  }

  std::unique_ptr<ast::BooleanLiteral> Bool = parseBooleanLiteral(S);
  if (Bool) {
    return Bool;
  }

  std::unique_ptr<ast::Identifier> Id = parseIdentifier(S);
  if (Id) {
    return Id;
  }

  std::unique_ptr<ast::Values> V = parseValues(S);
  if (V) {
    return V;
  }

  std::unique_ptr<ast::Lambda> L = parseLambda(S);
  if (L) {
    return L;
  }

  std::unique_ptr<ast::Begin> B = parseBegin(S);
  if (B) {
    return B;
  }

  std::unique_ptr<ast::SetBang> SB = parseSetBang(S);
  if (SB) {
    return SB;
  }

  std::unique_ptr<ast::IfCond> IC = parseIfCond(S);
  if (IC) {
    return IC;
  }

  std::unique_ptr<ast::LetValues> LV = parseLetValues(S);
  if (LV) {
    return LV;
  }

  std::unique_ptr<ast::Application> A = parseApplication(S);
  if (A) {
    return A;
  }

  return nullptr;
}

std::unique_ptr<ast::Identifier> parseIdentifier(Stream &S) {
  Tok IDTok = gettok(S);
  if (IDTok.tok != Tok::TokType::ID) {
    S.rewind(IDTok);
    return nullptr;
  }
  return std::make_unique<ast::Identifier>(
      IdPool::instance().create(IDTok.value));
}

// Parses an expression of the form:
// (define-values (id ...) expr)
std::unique_ptr<ast::DefineValues> parseDefineValues(Stream &S) {
  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewind(T);
    return nullptr;
  }

  Tok DVTok = gettok(S);
  if (DVTok.tok != Tok::TokType::DEFINE_VALUES) {
    S.rewind(DVTok);
    S.rewind(T);
    return nullptr;
  }

  // parse the list of ids
  std::vector<ast::Identifier> Ids;
  T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    return nullptr;
  }

  while (true) {
    std::unique_ptr<ast::Identifier> Id = parseIdentifier(S);
    if (!Id) {
      break;
    }
    Ids.emplace_back(*Id);
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    return nullptr;
  }

  // parse the expression
  std::unique_ptr<ast::ExprNode> Expr = parseExpr(S);
  if (!Expr) {
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    return nullptr;
  }

  return std::make_unique<ast::DefineValues>(Ids, Expr);
}

// Parses an expression of the form:
// (values expr ...)
std::unique_ptr<ast::Values> parseValues(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  std::optional<Tok> IDTok = maybeLexIdOrNumber(S);
  if (!IDTok || IDTok->tok != Tok::TokType::VALUES) {
    S.rewindTo(Start);
    return nullptr;
  }

  std::vector<std::unique_ptr<ast::ExprNode>> Exprs;
  while (true) {
    std::unique_ptr<ast::ExprNode> Expr = parseExpr(S);
    if (!Expr) {
      break;
    }
    Exprs.push_back(std::move(Expr));
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    return nullptr;
  }

  return std::make_unique<ast::Values>(std::move(Exprs));
}

std::unique_ptr<ast::Linklet> parseLinklet(Stream &S) {
  //   (linklet [[imported-id/renamed ...] ...]
  //          [exported-id/renamed ...]
  //   defn-or-expr ...)
  //
  // imported-id/renamed	 	=	 	imported-id
  //  	 	|	 	(external-imported-id
  //  internal-imported-id)
  //
  // exported-id/renamed	 	=	 	exported-id
  //  	 	|	 	(internal-exported-id
  //  external-exported-id)

  size_t Start = S.getPosition();
  auto Linklet = std::make_unique<ast::Linklet>();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::LINKLET) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  /// FIXME: Add imports support
  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  // Parsing linklet exports
  const std::vector<ast::Linklet::idpair_t> Exports = parseLinkletExports(S);
  for (const auto &E : Exports) {
    Linklet->appendExport(E.first, E.second);
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  while (true) {
    std::unique_ptr<ast::TLNode> Exp = parseDefnOrExpr(S);
    if (!Exp) {
      break;
    }
    Linklet->appendBodyForm(std::move(Exp));
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  if (Linklet->bodyFormsCount() == 0) {
    S.rewindTo(Start);
    return nullptr;
  }

  return Linklet;
}

// Parse lambda expression of the form:
// (lambda <formals> body)
// where formals are parsed by parseFormals
std::unique_ptr<ast::Lambda> parseLambda(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::LAMBDA) {
    S.rewindTo(Start);
    return nullptr;
  }

  // Lets create the lambda node
  std::unique_ptr<ast::Lambda> Lambda = std::make_unique<ast::Lambda>();

  std::unique_ptr<ast::Formal> Formals = parseFormals(S);
  if (!Formals) {
    S.rewindTo(Start);
    return nullptr;
  }
  Lambda->setFormals(std::move(Formals));

  std::unique_ptr<ast::ExprNode> Body = parseExpr(S);
  if (!Body) {
    S.rewindTo(Start);
    return nullptr;
  }
  Lambda->setBody(std::move(Body));

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  return Lambda;
}

// Parse formals of the form:
// (id ...)
// (id ... . id)
// id
std::unique_ptr<ast::Formal> parseFormals(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewind(T);

    // If it's not a list, it's a single id
    std::unique_ptr<ast::Identifier> Id = parseIdentifier(S);
    if (Id) {
      return std::make_unique<ast::IdentifierFormal>(*Id);
    }

    S.rewindTo(Start);
    return nullptr;
  }

  std::vector<ast::Identifier> Ids;
  while (true) {
    T = gettok(S);
    if (T.tok == Tok::TokType::RPAREN || T.tok == Tok::TokType::DOT) {
      break;
    }
    S.rewind(T); // put the token back

    std::unique_ptr<ast::Identifier> Id = parseIdentifier(S);
    if (!Id) {
      S.rewindTo(Start);
      return nullptr;
    }

    Ids.push_back(*Id);
  }

  if (T.tok == Tok::TokType::DOT) {

    std::unique_ptr<ast::Identifier> RestId = parseIdentifier(S);
    if (!RestId) {
      S.rewindTo(Start);
      return nullptr;
    }

    T = gettok(S);
    if (T.tok != Tok::TokType::RPAREN) {
      S.rewindTo(Start);
      return nullptr;
    }

    return std::make_unique<ast::ListRestFormal>(Ids, *RestId);
  }

  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  return std::make_unique<ast::ListFormal>(Ids);
}

// Parse lambda expression of the form:
// (begin <expr>+) | (begin0 <expr>+)
std::unique_ptr<ast::Begin> parseBegin(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::BEGIN && T.tok != Tok::TokType::BEGIN0) {
    S.rewindTo(Start);
    return nullptr;
  }

  // Lets create the begin node
  std::unique_ptr<ast::Begin> Begin = std::make_unique<ast::Begin>();
  if (T.tok == Tok::TokType::BEGIN0) {
    Begin->markAsBegin0();
  }

  while (true) {
    std::unique_ptr<ast::ExprNode> Exp = parseExpr(S);
    if (!Exp) {
      break;
    }
    Begin->appendExpr(std::move(Exp));
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  if (Begin->bodyCount() == 0) {
    S.rewindTo(Start);
    return nullptr;
  }

  return Begin;
}

// Parse application of the form:
// (<expr> <expr>*)
std::unique_ptr<ast::Application> parseApplication(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  auto App = std::make_unique<ast::Application>();

  while (true) {
    std::unique_ptr<ast::ExprNode> Exp = parseExpr(S);
    if (!Exp) {
      break;
    }
    App->appendExpr(std::move(Exp));
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  if (App->length() == 0) {
    S.rewindTo(Start);
    return nullptr;
  }

  return App;
}

// Parse set!:
// (set! <id> <expr>)
std::unique_ptr<ast::SetBang> parseSetBang(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::SETBANG) {
    S.rewindTo(Start);
    return nullptr;
  }

  auto Set = std::make_unique<ast::SetBang>();

  std::unique_ptr<ast::Identifier> Id = parseIdentifier(S);
  if (!Id) {
    S.rewindTo(Start);
    return nullptr;
  }
  Set->setIdentifier(std::move(Id));

  std::unique_ptr<ast::ExprNode> Exp = parseExpr(S);
  if (!Exp) {
    S.rewindTo(Start);
    return nullptr;
  }
  Set->setExpr(std::move(Exp));

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  return Set;
}

std::unique_ptr<ast::IfCond> parseIfCond(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::IF) {
    S.rewindTo(Start);
    return nullptr;
  }

  auto If = std::make_unique<ast::IfCond>();

  std::unique_ptr<ast::ExprNode> Cond = parseExpr(S);
  if (!Cond) {
    S.rewindTo(Start);
    return nullptr;
  }
  If->setCond(std::move(Cond));

  std::unique_ptr<ast::ExprNode> Then = parseExpr(S);
  if (!Then) {
    S.rewindTo(Start);
    return nullptr;
  }
  If->setThen(std::move(Then));

  std::unique_ptr<ast::ExprNode> Else = parseExpr(S);
  if (!Else) {
    S.rewindTo(Start);
    return nullptr;
  }
  If->setElse(std::move(Else));

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  return If;
}

// Parse boolean literal
std::unique_ptr<ast::BooleanLiteral> parseBooleanLiteral(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok == Tok::TokType::BOOL_TRUE) {
    return std::make_unique<ast::BooleanLiteral>(true);
  }
  if (T.tok == Tok::TokType::BOOL_FALSE) {
    return std::make_unique<ast::BooleanLiteral>(false);
  }
  S.rewindTo(Start);

  return nullptr;
}

// Parse a let-values form.
// (let-values ([(id ...) val-expr] ...) body ...+)
std::unique_ptr<ast::LetValues> parseLetValues(Stream &S) {
  size_t Start = S.getPosition();

  Tok T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::LET_VALUES) {
    S.rewindTo(Start);
    return nullptr;
  }

  auto Let = std::make_unique<ast::LetValues>();

  T = gettok(S);
  if (T.tok != Tok::TokType::LPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  while (true) {
    T = gettok(S);
    if (T.tok == Tok::TokType::RPAREN) {
      break;
    }

    // Parse the binding.
    if (T.tok != Tok::TokType::LPAREN) {
      S.rewindTo(Start);
      return nullptr;
    }

    // Parse list of identifiers.
    T = gettok(S);
    if (T.tok != Tok::TokType::LPAREN) {
      S.rewindTo(Start);
      return nullptr;
    }

    std::vector<ast::Identifier> Ids;
    while (true) {
      T = gettok(S);
      if (T.tok == Tok::TokType::RPAREN) {
        break;
      }

      S.rewind(T);
      std::unique_ptr<ast::Identifier> Id = parseIdentifier(S);
      if (!Id) {
        S.rewindTo(Start);
        return nullptr;
      }
      Ids.push_back(*Id);
    }

    // Parse the value expression.
    std::unique_ptr<ast::ExprNode> Val = parseExpr(S);
    if (!Val) {
      S.rewindTo(Start);
      return nullptr;
    }

    T = gettok(S);
    if (T.tok != Tok::TokType::RPAREN) {
      S.rewindTo(Start);
      return nullptr;
    }

    Let->appendBinding(std::move(Ids), std::move(Val));
  }

  while (true) {
    std::unique_ptr<ast::ExprNode> Exp = parseExpr(S);
    if (!Exp) {
      break;
    }
    Let->appendBody(std::move(Exp));
  }

  T = gettok(S);
  if (T.tok != Tok::TokType::RPAREN) {
    S.rewindTo(Start);
    return nullptr;
  }

  if (Let->bodyCount() == 0) {
    S.rewindTo(Start);
    return nullptr;
  }

  return Let;
}