
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "Diagnostics.h"
#include "Lex.h"
#include "Parse.h"
#include "SourceStream.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <optional>
#include <string>

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

TEST_CASE("Lexing Keyword tokens", "[parser]") {
  SourceStream Kw("#:foo");
  Tok Tok = gettok(Kw);
  REQUIRE(Tok.is(Tok::TokType::KEYWORD));
  REQUIRE(Tok.Value == "foo");

  SourceStream QuotedKw("'#:bar");
  Tok = gettok(QuotedKw);
  REQUIRE(Tok.is(Tok::TokType::SYMBOLMARK));
  Tok = gettok(QuotedKw);
  REQUIRE(Tok.is(Tok::TokType::KEYWORD));
  REQUIRE(Tok.Value == "bar");
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

TEST_CASE("Lexing letrec-values", "[parser]") {
  SourceStream Letvals("(letrec-values () #f)");
  Tok Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::LPAREN));
  Tok = gettok(Letvals);
  REQUIRE(Tok.is(Tok::TokType::LETREC_VALUES));
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

TEST_CASE("Parsing case-lambda", "[parser]") {
  SourceStream CL1("(case-lambda ((x) x) ((x y) (+ x y)))");
  std::unique_ptr<ast::CaseLambda> CL = parseCaseLambda(CL1);
  REQUIRE(CL);
  REQUIRE(CL->size() == 2);
  REQUIRE((*CL)[0].getFormalsType() == ast::Formal::Type::List);
  REQUIRE((*CL)[1].getFormalsType() == ast::Formal::Type::List);

  // A clause may use a rest formal.
  SourceStream CL2("(case-lambda ((x) x) ((x . y) y))");
  CL = parseCaseLambda(CL2);
  REQUIRE(CL);
  REQUIRE(CL->size() == 2);
  REQUIRE((*CL)[1].getFormalsType() == ast::Formal::Type::ListRest);

  // A case-lambda with no clauses is valid.
  SourceStream CL3("(case-lambda)");
  CL = parseCaseLambda(CL3);
  REQUIRE(CL);
  REQUIRE(CL->size() == 0);
}

TEST_CASE("Parsing begin", "[parser]") {
  SourceStream B1("(begin 1 2 3)");
  std::unique_ptr<ast::Begin> B = parseBegin(B1);

  REQUIRE(B);
  REQUIRE(B->bodyCount() == 3);
}

TEST_CASE("DiagnosticEngine counts diagnostics", "[diagnostics]") {
  nora::DiagnosticEngine Diag;
  REQUIRE_FALSE(Diag.hadError());
  REQUIRE(Diag.getNumErrors() == 0);
  REQUIRE(Diag.getNumWarnings() == 0);

  Diag.error("first failure");
  Diag.error("second failure");
  REQUIRE(Diag.hadError());
  REQUIRE(Diag.getNumErrors() == 2);
  REQUIRE(Diag.getNumWarnings() == 0);
}

TEST_CASE("SourceStream getLoc maps offsets into the buffer", "[diagnostics]") {
  nora::DiagnosticEngine Diag;
  SourceStream S("(+ 1 2)", &Diag);

  llvm::SMLoc L0 = S.getLoc(0);
  llvm::SMLoc L3 = S.getLoc(3);
  REQUIRE(L0.isValid());
  REQUIRE(L3.isValid());
  REQUIRE(L3.getPointer() - L0.getPointer() == 3);
}

TEST_CASE("DiagnosticEngine renders line and column", "[diagnostics]") {
  nora::DiagnosticEngine Diag;
  SourceStream S("(+ 1 2)", &Diag);

  std::string Buf;
  llvm::raw_string_ostream OS(Buf);
  // Offset 3 is the digit '1' -> line 1, column 4.
  Diag.getSourceMgr().PrintMessage(OS, S.getLoc(3), llvm::SourceMgr::DK_Error,
                                   "boom");
  OS.flush();
  REQUIRE(Buf.find("error: boom") != std::string::npos);
  REQUIRE(Buf.find("1:4") != std::string::npos);
}

TEST_CASE("Parsing variable references", "[parser]") {
  SourceStream V1("(#%variable-reference)");
  std::unique_ptr<ast::VariableReference> V = parseVariableReference(V1);
  REQUIRE(V);
  REQUIRE_FALSE(V->hasId());

  SourceStream V2("(#%variable-reference x)");
  V = parseVariableReference(V2);
  REQUIRE(V);
  REQUIRE(V->hasId());
  REQUIRE(V->getId().getName() == "x");

  SourceStream V3("(#%variable-reference (#%top . x))");
  V = parseVariableReference(V3);
  REQUIRE(V);
  REQUIRE(V->hasId());
  REQUIRE(V->getId().getName() == "x");

  // A plain application must not be misparsed as a variable reference.
  SourceStream V4("(f x)");
  V = parseVariableReference(V4);
  REQUIRE_FALSE(V);
}