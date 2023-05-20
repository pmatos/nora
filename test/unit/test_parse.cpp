
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "Lex.h"
#include "Parse.h"

#include <llvm/Support/Casting.h>

#include <optional>

using namespace Lex;
using namespace Parse;

// TESTS
TEST_CASE("Lexing Number or Identifier Tokens", "[parser]") {
  std::optional<Tok> T;

  SourceStream Stream1("12412abc");
  T = maybeLexIdOrNumber(Stream1);
  REQUIRE(T);
  REQUIRE(T.value().is(Tok::TokType::ID));
  REQUIRE(T.value().Value == "12412abc");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 7);

  SourceStream Stream2("-41233()");
  T = maybeLexIdOrNumber(Stream2);
  REQUIRE(T);
  REQUIRE(T.value().is(Tok::TokType::NUM));
  REQUIRE(T.value().Value == "-41233");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 5);

  SourceStream Stream3("-1 asdv");
  T = maybeLexIdOrNumber(Stream3);
  REQUIRE(T);
  REQUIRE(T.value().is(Tok::TokType::NUM));
  REQUIRE(T.value().Value == "-1");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 1);

  SourceStream Stream4("0 ()");
  T = maybeLexIdOrNumber(Stream4);
  REQUIRE(T);
  REQUIRE(T.value().is(Tok::TokType::NUM));
  REQUIRE(T.value().Value == "0");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 0);

  SourceStream Stream5("a1234");
  T = maybeLexIdOrNumber(Stream5);
  REQUIRE(T);
  REQUIRE(T.value().is(Tok::TokType::ID));
  REQUIRE(T.value().Value == "a1234");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 4);

  SourceStream Stream6("-a");
  T = maybeLexIdOrNumber(Stream6);
  REQUIRE(T);
  REQUIRE(T.value().is(Tok::TokType::ID));
  REQUIRE(T.value().Value == "-a");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 1);

  SourceStream Stream7("hello");
  T = maybeLexIdOrNumber(Stream7);
  REQUIRE(T);
  REQUIRE(T.value().Value == "hello");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 4);

  SourceStream Stream8("hello world");
  T = maybeLexIdOrNumber(Stream8);
  REQUIRE(T);
  REQUIRE(T.value().Value == "hello");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 4);

  SourceStream Stream9("hello(world)");
  T = maybeLexIdOrNumber(Stream9);
  REQUIRE(T);
  REQUIRE(T.value().Value == "hello");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 4);

  SourceStream Stream10("1/bound-identifier=? bound-identifier=?");
  T = maybeLexIdOrNumber(Stream10);
  REQUIRE(T);
  REQUIRE(T.value().Value == "1/bound-identifier=?");
  REQUIRE(T.value().Start == 0);
  REQUIRE(T.value().End == 19);

  SourceStream Stream11(
      "(if (variable-reference-from-unsafe? (#%variable-reference)) "
      "(void)(void)) ");
  Tok Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::LPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::IF));
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::LPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.Value == "variable-reference-from-unsafe?");
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::LPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.Value == "#%variable-reference");
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::RPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::RPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::LPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.Value == "void");
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::RPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::LPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.Value == "void");
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::RPAREN));
  Ts = gettok(Stream11);
  REQUIRE(Ts.is(Tok::TokType::RPAREN));

  Tok Tok;
  SourceStream Stream12("2");
  Tok = gettok(Stream12);
  REQUIRE(Tok.is(Tok::TokType::NUM));
  REQUIRE(Tok.Value == "2");
}

TEST_CASE("Lexing String tokens", "[parser]") {
  SourceStream Str1(R"("Hello World")");
  Tok Tok = gettok(Str1);
  REQUIRE(Tok.is(Tok::TokType::STRING));
  REQUIRE(Tok.Value == R"("Hello World")");
  SourceStream Str2(R"("hello \"but also this\"")");
  Tok = gettok(Str2);
  REQUIRE(Tok.is(Tok::TokType::STRING));
  REQUIRE(Tok.Value == R"("hello \"but also this\"")");
  SourceStream Str3(R"("\\")");
  Tok = gettok(Str3);
  REQUIRE(Tok.is(Tok::TokType::STRING));
  REQUIRE(Tok.Value == R"("\\")");
  SourceStream Str4(R"("Hello World")");
  Tok = gettok(Str4);
  REQUIRE(Tok.is(Tok::TokType::STRING));
  REQUIRE(Tok.Value == R"("Hello World")");

  SourceStream Stream5(R"(#"I am a byte string.")");
  Tok = gettok(Stream5);
  REQUIRE(Tok.is(Tok::TokType::BYTE_STRING));
  REQUIRE(Tok.Value == R"("I am a byte string.")");
}

TEST_CASE("Lexing Symbol tokens", "[parser]") {
  SourceStream Sym("'racket-sym");
  Tok Tok = gettok(Sym);
  REQUIRE(Tok.is(Tok::TokType::SYMBOLMARK));
  Tok = gettok(Sym);
  REQUIRE(Tok.is(Tok::TokType::ID));
  REQUIRE(Tok.Value == "racket-sym");

  SourceStream Sym1("'  foo");
  Tok = gettok(Sym1);
  REQUIRE(Tok.is(Tok::TokType::SYMBOLMARK));
  Tok = gettok(Sym1);
  REQUIRE(Tok.is(Tok::TokType::ID));
  REQUIRE(Tok.Value == "foo");

  SourceStream Sym2(R"('#\\)");
  Tok = gettok(Sym2);
  REQUIRE(Tok.is(Tok::TokType::SYMBOLMARK));
  Tok = gettok(Sym2);
  REQUIRE(Tok.is(Tok::TokType::CHAR));
  REQUIRE(Tok.Value == R"(\)");
}

TEST_CASE("Lexing identifiers starting with numbers", "[parser]") {
  SourceStream Sym("1/bound-identifier=?");
  Tok Tok = gettok(Sym);
  REQUIRE(Tok.is(Tok::TokType::ID));
  REQUIRE(Tok.Value == "1/bound-identifier=?");
}

TEST_CASE("Lexing booleans", "[parser]") {
  SourceStream Boolt("#t");
  Tok Tok = gettok(Boolt);
  REQUIRE(Tok.is(Tok::TokType::BOOL_TRUE));
  SourceStream Boolf("#f 2");
  Tok = gettok(Boolf);
  REQUIRE(Tok.is(Tok::TokType::BOOL_FALSE));
}

TEST_CASE("Lexing full expressions", "[parser]") {
  SourceStream Lambda("(lambda () (void))");
  Tok Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::LPAREN));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::LAMBDA));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::LPAREN));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::RPAREN));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::LPAREN));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::VOID));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::RPAREN));
  Tok = gettok(Lambda);
  REQUIRE(Tok.is(Tok::TokType::RPAREN));
}

TEST_CASE("Lexing full expressions 2", "[parser]") {
  SourceStream Letvals("(let-values () #f)");
  Tok Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::LPAREN));
  Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::LET_VALUES));
  Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::LPAREN));
  Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::RPAREN));
  Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::BOOL_FALSE));
  Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::RPAREN));
}

TEST_CASE("Lexing Characters", "[parser]") {
  SourceStream Char1(R"(#\space)");
  Tok Tok = gettok(Char1);
  REQUIRE(Tok.is(Tok::TokType::CHAR_NAMED));
  REQUIRE(Tok.Value == R"(space)");

  SourceStream Char2(R"(#\u10ff)");
  Tok = gettok(Char2);
  REQUIRE(Tok.is(Tok::TokType::CHAR_HEX));
  REQUIRE(Tok.Value == R"(u10ff)");
}

TEST_CASE("Lexing Regex Literals", "[parser]") {
  Tok Tok;

  SourceStream Stream1(R"(#rx"^[\\][\\][?][\\]")");
  Tok = gettok(Stream1);
  REQUIRE(Tok.is(Tok::TokType::REGEXP_LITERAL));
  REQUIRE(Tok.Value == R"("^[\\][\\][?][\\]")");

  SourceStream Stream2(R"(#rx#"\0")");
  Tok = gettok(Stream2);
  REQUIRE(Tok.is(Tok::TokType::BYTE_REGEXP_LITERAL));
  REQUIRE(Tok.Value == R"("\0")");
}

TEST_CASE("Parsing linklets", "[parser]") {
  SourceStream Linklet("(linklet () () 2)");
  std::unique_ptr<ast::Linklet> L = parseLinklet(Linklet);
  REQUIRE(L);

  REQUIRE(L->exportsCount() == 0);
  REQUIRE(L->importsCount() == 0);

  const auto &Body = L->getBody();
  const ast::TLNode &Node = Body[0];
  REQUIRE(llvm::isa<ast::Integer>(Node));
  const auto &Integer = llvm::cast<ast::Integer>(Node);
  REQUIRE(Integer == 2);
}

TEST_CASE("Parsing lambdas", "[parser]") {

  SourceStream L1("(lambda () 2)");
  std::unique_ptr<ast::Lambda> L = parseLambda(L1);
  REQUIRE(L);

  REQUIRE(L->getFormalsType() == ast::Formal::Type::List);
  const ast::ExprNode &B = L->getBody();
  const auto &Integer = llvm::cast<ast::Integer>(B);
  REQUIRE(Integer == 2);

  SourceStream L2("(lambda (x y) x)");
  L = parseLambda(L2);
  REQUIRE(L);

  REQUIRE(L->getFormalsType() == ast::Formal::Type::List);
  const ast::ExprNode &B2 = L->getBody();
  const auto &Var = llvm::cast<ast::Identifier>(B2);
  REQUIRE(Var.getName() == "x");
}

TEST_CASE("Parsing begin", "[parser]") {
  SourceStream B1("(begin 1 2 3)");
  std::unique_ptr<ast::Begin> B = parseBegin(B1);

  REQUIRE(B);
  REQUIRE(B->bodyCount() == 3);
}