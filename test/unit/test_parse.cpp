
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "ast/arithplus.h"
#include "ast/definevalues.h"
#include "ast/identifier.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/linklet.h"
#include "ast/values.h"
#include "ast/void.h"
#include "parse.h"
#include "toplevelnode.h"
#include "utils/idpool.h"
#include <optional>

// TESTS
TEST_CASE("Lexing Number or Identifier Tokens", "[parser]") {
  std::optional<Tok> T;

  Stream Stream1(L"12412abc");
  T = maybeLexIdOrNumber(Stream1);
  REQUIRE(T);
  REQUIRE(T.value().tok == Tok::TokType::ID);
  REQUIRE(T.value().value == L"12412abc");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 7);

  Stream Stream2(L"-41233()");
  T = maybeLexIdOrNumber(Stream2);
  REQUIRE(T);
  REQUIRE(T.value().tok == Tok::TokType::NUM);
  REQUIRE(T.value().value == L"-41233");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 5);

  Stream Stream3(L"-1 asdv");
  T = maybeLexIdOrNumber(Stream3);
  REQUIRE(T);
  REQUIRE(T.value().tok == Tok::TokType::NUM);
  REQUIRE(T.value().value == L"-1");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 1);

  Stream Stream4(L"0 ()");
  T = maybeLexIdOrNumber(Stream4);
  REQUIRE(T);
  REQUIRE(T.value().tok == Tok::TokType::NUM);
  REQUIRE(T.value().value == L"0");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 0);

  Stream Stream5(L"a1234");
  T = maybeLexIdOrNumber(Stream5);
  REQUIRE(T);
  REQUIRE(T.value().tok == Tok::TokType::ID);
  REQUIRE(T.value().value == L"a1234");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 4);

  Stream Stream6(L"-a");
  T = maybeLexIdOrNumber(Stream6);
  REQUIRE(T);
  REQUIRE(T.value().tok == Tok::TokType::ID);
  REQUIRE(T.value().value == L"-a");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 1);

  Stream Stream7(L"hello");
  T = maybeLexIdOrNumber(Stream7);
  REQUIRE(T);
  REQUIRE(T.value().value == L"hello");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 4);

  Stream Stream8(L"hello world");
  T = maybeLexIdOrNumber(Stream8);
  REQUIRE(T);
  REQUIRE(T.value().value == L"hello");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 4);

  Stream Stream9(L"hello(world)");
  T = maybeLexIdOrNumber(Stream9);
  REQUIRE(T);
  REQUIRE(T.value().value == L"hello");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 4);

  Stream Stream10(L"1/bound-identifier=? bound-identifier=?");
  T = maybeLexIdOrNumber(Stream10);
  REQUIRE(T);
  REQUIRE(T.value().value == L"1/bound-identifier=?");
  REQUIRE(T.value().start == 0);
  REQUIRE(T.value().end == 19);

  Stream Stream11(
      L"(if (variable-reference-from-unsafe? (#%variable-reference)) "
      "(void)(void)) ");
  Tok Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::LPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::IF);
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::LPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.value == L"variable-reference-from-unsafe?");
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::LPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.value == L"#%variable-reference");
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::RPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::RPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::LPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.value == L"void");
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::RPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::LPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.value == L"void");
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::RPAREN);
  Ts = gettok(Stream11);
  REQUIRE(Ts.tok == Tok::TokType::RPAREN);

  Tok Tok;
  Stream Stream12(L"2");
  Tok = gettok(Stream12);
  REQUIRE(Tok.tok == Tok::TokType::NUM);
  REQUIRE(Tok.value == L"2");
}

TEST_CASE("Lexing String tokens", "[parser]") {
  Stream str1(LR"("Hello World")");
  Tok Tok = gettok(str1);
  REQUIRE(Tok.tok == Tok::TokType::STRING);
  REQUIRE(Tok.value == LR"("Hello World")");
  Stream str2(LR"("hello \"but also this\"")");
  Tok = gettok(str2);
  REQUIRE(Tok.tok == Tok::TokType::STRING);
  REQUIRE(Tok.value == LR"("hello \"but also this\"")");
  Stream str3(LR"("\\")");
  Tok = gettok(str3);
  REQUIRE(Tok.tok == Tok::TokType::STRING);
  REQUIRE(Tok.value == LR"("\\")");
  Stream str4(LR"("Hello World")");
  Tok = gettok(str4);
  REQUIRE(Tok.tok == Tok::TokType::STRING);
  REQUIRE(Tok.value == LR"("Hello World")");

  Stream Stream5(LR"(#"I am a byte string.")");
  Tok = gettok(Stream5);
  REQUIRE(Tok.tok == Tok::TokType::BYTE_STRING);
  REQUIRE(Tok.value == LR"("I am a byte string.")");
}

TEST_CASE("Lexing Symbol tokens", "[parser]") {
  Stream sym(L"'racket-sym");
  Tok Tok = gettok(sym);
  REQUIRE(Tok.tok == Tok::TokType::SYMBOLMARK);
  Tok = gettok(sym);
  REQUIRE(Tok.tok == Tok::TokType::ID);
  REQUIRE(Tok.value == L"racket-sym");

  Stream sym1(L"'  foo");
  Tok = gettok(sym1);
  REQUIRE(Tok.tok == Tok::TokType::SYMBOLMARK);
  Tok = gettok(sym1);
  REQUIRE(Tok.tok == Tok::TokType::ID);
  REQUIRE(Tok.value == L"foo");

  Stream sym2(LR"('#\\)");
  Tok = gettok(sym2);
  REQUIRE(Tok.tok == Tok::TokType::SYMBOLMARK);
  Tok = gettok(sym2);
  REQUIRE(Tok.tok == Tok::TokType::CHAR);
  REQUIRE(Tok.value == LR"(\)");
}

TEST_CASE("Lexing identifiers starting with numbers", "[parser]") {
  Stream sym(L"1/bound-identifier=?");
  Tok Tok = gettok(sym);
  REQUIRE(Tok.tok == Tok::TokType::ID);
  REQUIRE(Tok.value == L"1/bound-identifier=?");
}

TEST_CASE("Lexing booleans", "[parser]") {
  Stream boolt(L"#t");
  Tok Tok = gettok(boolt);
  REQUIRE(Tok.tok == Tok::TokType::BOOL_TRUE);
  Stream boolf(L"#f 2");
  Tok = gettok(boolf);
  REQUIRE(Tok.tok == Tok::TokType::BOOL_FALSE);
}

TEST_CASE("Lexing full expressions", "[parser]") {
  Stream lambda(L"(lambda () (void))");
  Tok Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::LPAREN);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::LAMBDA);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::LPAREN);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::RPAREN);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::LPAREN);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::VOID);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::RPAREN);
  Tok = gettok(lambda);
  REQUIRE(Tok.tok == Tok::TokType::RPAREN);
}

TEST_CASE("Lexing full expressions 2", "[parser]") {
  Stream letvals(L"(let-values () #f)");
  Tok Tok = gettok(letvals);
  REQUIRE(Tok.tok == Tok::TokType::LPAREN);
  Tok = gettok(letvals);
  REQUIRE(Tok.tok == Tok::TokType::LET_VALUES);
  Tok = gettok(letvals);
  REQUIRE(Tok.tok == Tok::TokType::LPAREN);
  Tok = gettok(letvals);
  REQUIRE(Tok.tok == Tok::TokType::RPAREN);
  Tok = gettok(letvals);
  REQUIRE(Tok.tok == Tok::TokType::BOOL_FALSE);
  Tok = gettok(letvals);
  REQUIRE(Tok.tok == Tok::TokType::RPAREN);
}

TEST_CASE("Lexing Characters", "[parser]") {
  Stream char1(LR"(#\space)");
  Tok Tok = gettok(char1);
  REQUIRE(Tok.tok == Tok::TokType::CHAR_NAMED);
  REQUIRE(Tok.value == LR"(space)");

  Stream char2(LR"(#\u10ff)");
  Tok = gettok(char2);
  REQUIRE(Tok.tok == Tok::TokType::CHAR_HEX);
  REQUIRE(Tok.value == LR"(u10ff)");
}

TEST_CASE("Lexing Regex Literals", "[parser]") {
  Tok Tok;

  Stream Stream1(LR"(#rx"^[\\][\\][?][\\]")");
  Tok = gettok(Stream1);
  REQUIRE(Tok.tok == Tok::TokType::REGEXP_LITERAL);
  REQUIRE(Tok.value == LR"("^[\\][\\][?][\\]")");

  Stream Stream2(LR"(#rx#"\0")");
  Tok = gettok(Stream2);
  REQUIRE(Tok.tok == Tok::TokType::BYTE_REGEXP_LITERAL);
  REQUIRE(Tok.value == LR"("\0")");
}

TEST_CASE("Parsing linklets", "[parser]") {
  Stream Linklet(L"(linklet () () 2)");
  std::unique_ptr<nir::Linklet> L = parseLinklet(Linklet);
  REQUIRE(L);

  REQUIRE(L->exportsCount() == 0);
  REQUIRE(L->importsCount() == 0);

  const nir::TLNode &B = L->getBody()[0];
  const auto &Integer = std::get<nir::Integer>(B);
  REQUIRE(Integer == 2);
}

TEST_CASE("Parsing lambdas", "[parser]") {

  Stream L1(L"(lambda () 2)");
  std::unique_ptr<nir::Lambda> L = parseLambda(L1);
  REQUIRE(L);

  REQUIRE(L->getFormalsType() == nir::Formal::Type::List);
  const nir::ExprNode &B = L->getBody();
  const auto &Integer = std::get<nir::Integer>(B);
  REQUIRE(Integer == 2);

  Stream L2(L"(lambda (x y) x)");
  L = parseLambda(L2);
  REQUIRE(L);

  REQUIRE(L->getFormalsType() == nir::Formal::Type::List);
  const nir::ExprNode &B2 = L->getBody();
  const auto &Var = std::get<nir::Identifier>(B2);
  REQUIRE(Var == IdPool::instance().create(L"x"));
}