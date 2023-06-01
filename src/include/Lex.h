#pragma once

#include "SourceStream.h"

#include <optional>
#include <string>
#include <string_view>

namespace Lex {

class Tok {
public:
  // Available token types
  enum TokType {
    INVALID,
    DOT,
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

  Tok() : Token(INVALID), Value(), Start(0), End(0) {}
  Tok(TokType T, size_t Pos) : Token(T), Value(), Start(Pos), End(Pos) {}
  Tok(TokType T, size_t Start, size_t End)
      : Token(T), Value(), Start(Start), End(End) {}
  Tok(TokType T, std::string_view Val, size_t Start, size_t End)
      : Token(T), Value(Val), Start(Start), End(End) {}

  friend std::ostream &operator<<(std::ostream &, const Tok &);
  size_t size() const { return End - Start + 1; }
  bool is(TokType T) const { return Token == T; }
  [[nodiscard]] bool isValid() const { return Token != TokType::INVALID; }

private:
  TokType Token;

public:
  // FIXME: make these private as well
  std::string Value;
  size_t Start; /// start position of token in stream
  size_t End;   /// end position of token in stream
};

Tok gettok(SourceStream &S);

std::optional<Tok> maybeLexIdOrNumber(SourceStream &S);
std::optional<Tok> maybeLexSymbolMark(SourceStream &S);
std::optional<Tok> maybeLexString(SourceStream &S);
std::optional<Tok> maybeLexSchemeChar(SourceStream &S);
std::optional<Tok> maybeLexBoolean(SourceStream &S);
std::optional<Tok> maybeLexHash(SourceStream &S);
std::optional<Tok> maybeLexKeyword(SourceStream &S);
std::optional<Tok> maybeLexRegexpLiteral(SourceStream &S);
std::optional<Tok> maybeLexVector(SourceStream &S);
std::optional<Tok> maybeLexDot(SourceStream &S);
std::optional<Tok> maybeLexString(SourceStream &S);
std::optional<Tok> maybeLexByteString(SourceStream &S);

bool isDelimiter(SourceStream &S, size_t Offset = 0);
bool isSymbolElement(SourceStream &S, size_t Offset = 0);
size_t isInlineHexEscape(SourceStream &S, size_t Offset = 0);
bool isMnemonicEscape(SourceStream &S, size_t Offset = 0);

}; // namespace Lex