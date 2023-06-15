#include "Lex.h"

#include "UTF8.h"

using namespace Lex;

std::ostream &operator<<(std::ostream &Out, const Tok &T) { return Out; }

// From:
// https://docs.racket-lang.org/reference/reader.html#%28part._parse-symbol%29
// which says (12/2022):
// A sequence that does not start with a delimiter or # is parsed as either a
// symbol, a number, or a extflonum, except that . by itself is never parsed
// as a symbol or number (unless the read-accept-dot parameter is set to #f).
// A successful number or extflonum parse takes precedence over a symbol
// parse. A #% also starts a symbol. The resulting symbol is interned. See the
// start of Delimiters and Dispatch for information about | and \ in parsing
// symbols.
std::optional<Tok> Lex::maybeLexIdOrNumber(SourceStream &S) {
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

    std::string_view Value = S.getSubviewAndSkip(Count);

    bool IsNumber = std::regex_match(Value.cbegin(), Value.cend(),
                                     std::regex("[-+]?[0-9]+"));
    return {Tok(IsNumber ? Tok::TokType::NUM : Tok::TokType::ID, Value, Start,
                Start + Count - 1)};
  }

  // or
  // just parse until we find a delimiter
  size_t Count = 0;
  while (!isDelimiter(S, Count) && S.peekChar(Count) != EOF &&
         !isspace(S.peekChar(Count))) {
    Count++;
  }

  // also cannot be a single dot
  if (Count == 0) {
    return {};
  }
  if (Count == 1 && S.peekChar() == '.') {
    return {};
  }

  std::string_view Value = S.getSubviewAndSkip(Count);
  bool IsNumber =
      std::regex_match(Value.cbegin(), Value.cend(), std::regex("[-+]?[0-9]+"));
  Tok T(IsNumber ? Tok::TokType::NUM : Tok::TokType::ID, Value, Start,
        Start + Count - 1);

  // Transform token
  std::array KnownTokens = {
      std::make_pair("lambda", Tok::TokType::LAMBDA),
      std::make_pair("quote", Tok::TokType::QUOTE),
      std::make_pair("define-values", Tok::TokType::DEFINE_VALUES),
      std::make_pair("values", Tok::TokType::VALUES),
      std::make_pair("linklet", Tok::TokType::LINKLET),
      std::make_pair("letrec-values", Tok::TokType::LETREC_VALUES),
      std::make_pair("begin", Tok::TokType::BEGIN),
      std::make_pair("begin0", Tok::TokType::BEGIN0),
      std::make_pair("set!", Tok::TokType::SETBANG),
      std::make_pair("case-lambda", Tok::TokType::CASE_LAMBDA),
      std::make_pair("if", Tok::TokType::IF),
      std::make_pair("void", Tok::TokType::VOID),
      std::make_pair("raise-argument-error",
                     Tok::TokType::RAISE_ARGUMENT_ERROR),
      std::make_pair("procedure-arity-includes/c",
                     Tok::TokType::PROCEDURE_ARITY_INCLUDES_C),
      std::make_pair("make-struct-type", Tok::TokType::MAKE_STRUCT_TYPE),
      std::make_pair("let-values", Tok::TokType::LET_VALUES),
      std::make_pair("with-continuation-mark",
                     Tok::TokType::WITH_CONTINUATION_MARK),
      std::make_pair("#%variable-reference",
                     Tok::TokType::K_VARIABLE_REFERENCE)};

  if (T.is(Tok::TokType::ID)) {
    for (auto [Name, Token] : KnownTokens) {
      if (T.Value == Name) {
        return {Tok(Token, T.Value, T.Start, T.End)};
      }
    }
  }

  return {T};
}

std::optional<Tok> Lex::maybeLexKeyword(SourceStream &S) {
  size_t Start = S.getPosition();
  if (!S.isPrefix("#:")) {
    return {};
  }
  S.skipPrefix(2);

  std::optional<Tok> Identifier = maybeLexIdOrNumber(S);
  if (!Identifier) {
    S.rewind(2);
    return {};
  }

  Tok T(Tok::TokType::KEYWORD, Identifier->Value, Start, Identifier->End);
  return {T};
}

std::optional<Tok> Lex::maybeLexVector(SourceStream &S) {
  size_t Start = S.getPosition();
  if (!S.isPrefix("#(")) {
    return {};
  }
  S.skipPrefix(2);

  Tok T(Tok::TokType::VECTOR_START, Start, S.getPosition());
  return {T};
}

std::optional<Tok> Lex::maybeLexBoolean(SourceStream &S) {
  if (S.isPrefix("#t") || S.isPrefix("#f")) {
    Tok::TokType Ty = S.peekChar(1) == L't' ? Tok::TokType::BOOL_TRUE
                                            : Tok::TokType::BOOL_FALSE;
    size_t Start = S.getPosition();
    size_t End = S.getPosition() + 1;
    auto Val = S.getSubviewAndSkip(2);
    Tok T(Ty, Val, Start, End);
    return {T};
  }

  return {};
}

std::optional<Tok> Lex::maybeLexHash(SourceStream &S) {

  if (S.isPrefix("#hasheqv(")) {
    Tok T(Tok::TokType::HASHEQV_START, S.getPosition(), S.getPosition() + 9);
    S.skipPrefix(9);
    return {T};
  }

  if (S.isPrefix("#hasheq(")) {
    Tok T(Tok::TokType::HASHEQ_START, S.getPosition(), S.getPosition() + 8);
    S.skipPrefix(8);
    return {T};
  }

  if (S.isPrefix("#hash(")) {
    Tok T(Tok::TokType::HASH_START, S.getPosition(), S.getPosition() + 6);
    S.skipPrefix(6);
    return {T};
  }

  return {};
}

std::optional<Tok> Lex::maybeLexDot(SourceStream &S) {
  if (S.isPrefix(".")) {
    Tok T(Tok::TokType::DOT, ".", S.getPosition(), S.getPosition());
    S.skipPrefix(1);
    return {T};
  }
  return {};
}

std::optional<Tok> Lex::maybeLexString(SourceStream &S) {
  std::cmatch CM;
  if (S.searchRegex(R"|("([^"\\]*(\\.[^"\\]*)*)")|", CM)) {
    Tok T(Tok::TokType::STRING, CM.str(), S.getPosition(),
          S.getPosition() + CM.length() - 1);
    S.skipPrefix(CM.length());
    return {T};
  }

  return {};
}

std::optional<Tok> Lex::maybeLexSymbolMark(SourceStream &S) {
  if (S.peekChar() != '\'') {
    return {};
  }

  Tok T(Tok::TokType::SYMBOLMARK, S.getPosition());
  S.skipPrefix(1);
  return T;
}

std::optional<Tok> Lex::maybeLexRegexpLiteral(SourceStream &S) {
  // Parses a byte regexp literal in racket
  // #rx# followed a by "-delimited string containing the pattern.
  size_t Start = S.getPosition();
  if (!S.isPrefix("#rx")) {
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
        Str->Value, Start, Str->End);
  return {T};
}

std::optional<Tok> Lex::maybeLexByteString(SourceStream &S) {
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

  Tok T(Tok::TokType::BYTE_STRING, Str->Value, Start, Str->End);
  return {T};
}

bool isCharacterName(SourceStream &S, std::string &Found) {
  static const std::array Names = {"space",     "newline", "alarm",
                                   "backspace", "delete",  "escape",
                                   "null",      "return",  "tab"};

  for (const auto *Name : Names) {
    if (S.isPrefix(Name)) {
      Found = Name;
      return true;
    }
  }
  return false;
}

std::optional<Tok> Lex::maybeLexSchemeChar(SourceStream &S) {
  size_t LexStart = S.getPosition();

  // A character looks like this
  // #\ followed by the represented char
  // #\u followed by the scalar value as hex number
  // #\<character name>
  if (!S.isPrefix("#\\")) {
    return {};
  }
  S.skipPrefix(2);

  std::string MaybeName;
  if (isCharacterName(S, MaybeName)) {
    std::string_view Value = S.getSubviewAndSkip(MaybeName.size());
    Tok T(Tok::TokType::CHAR_NAMED, Value, S.getPosition() - 2,
          S.getPosition());
    return {T};
  }

  std::cmatch CM;
  if (S.searchRegex(R"((?:u|x|U|X)[0-9a-fA-F]{4})", CM)) {
    Tok T(Tok::TokType::CHAR_HEX, CM.str(), S.getPosition() - 2,
          S.getPosition() + CM.length() - 1);
    S.skipPrefix(CM.length());
    return {T};
  }

  // A character is a literal wide character or the unicode replacement
  // character

  llvm::UTF8 Start = S.peekChar();
  unsigned Bytes = llvm::getNumBytesForUTF8(Start);
  assert(Bytes > 0 && "Invalid UTF8");
  llvm::StringRef Char = S.getSubview(Bytes);
  const llvm::UTF8 *StartPtr = (const llvm::UTF8 *)Char.data();
  const llvm::UTF8 *EndPtr = StartPtr + Bytes - 1;

  if ( // look at first character
      (UTF8::isUTF8ReplacementCharacter(StartPtr, EndPtr) ||
       UTF8::isGraphUTF8(StartPtr, EndPtr)) &&

      // followup character
      (isDelimiter(S, 1) || S.peekChar(1) == EOF ||
       std::isspace(S.peekChar(1)))) {
    auto Subview = S.getSubviewAndSkip(Bytes);
    Tok T(Tok::TokType::CHAR, Subview, LexStart, S.getPosition());
    return {T};
  }

  return {};
}

bool isLetter(SourceStream &S, size_t Offset) {
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

bool isSpecialInitial(SourceStream &S, size_t Offset) {
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

bool isInitial(SourceStream &S, size_t Offset) {
  if (isLetter(S, Offset) || isSpecialInitial(S, Offset)) {
    return true;
  }
  return false;
}

bool isDigit(SourceStream &S, size_t Offset) {
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

bool isExplicitSign(SourceStream &S, size_t Offset) {
  switch (S.peekChar(Offset)) {
  case '+':
  case '-':
    return true;
  default:
    return false;
  }
}

bool Lex::isSymbolElement(SourceStream &S, size_t Offset) {
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

bool isHexDigit(SourceStream &S, size_t Offset) {
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

size_t isHexScalarValue(SourceStream &S, size_t Offset) {
  size_t Count = 0;
  while (isHexDigit(S, Offset + Count)) {
    Count++;
  }
  return Count;
}

bool Lex::isMnemonicEscape(SourceStream &S, size_t Offset) {
  if (S.peekChar(Offset) == '\\' &&
      (S.peekChar(Offset + 1) == 'a' || S.peekChar(Offset + 1) == 'b' ||
       S.peekChar(Offset + 1) == 't' || S.peekChar(Offset + 1) == 'n' ||
       S.peekChar(Offset + 1) == 'r')) {
    return true;
  }
  return false;
}

size_t Lex::isInlineHexEscape(SourceStream &S, size_t Offset) {
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

bool Lex::isDelimiter(SourceStream &S, size_t Offset) {
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
Tok Lex::gettok(SourceStream &S) {
  // We start by skipping whitespace which is not part of a token
  S.skipWhitespace();

  // FIXME: we are allowing both [ and ] for readability but we don't really
  // count them to ensure they match so this is valid:
  // (+ 2 1]
  switch (S.peekChar()) {
  case '(':
  case '[':
    S.skipPrefix(1);
    return {Tok::TokType::LPAREN, S.getPosition()};
  case ')':
  case ']':
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