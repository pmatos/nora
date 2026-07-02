#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "AST.h"
#include "ASTRuntime.h"
#include "Diagnostics.h"
#include "Interpreter.h"
#include "Parse.h"
#include "SourceStream.h"

#include <llvm/Support/Casting.h>

#include <memory>
#include <string>

namespace {

// Parse + interpret a linklet source string at the interpreter's public seam.
struct Run {
  bool ok = false;                        // no diagnostics were reported
  std::unique_ptr<ast::ValueNode> result; // Interpreter::getResult()
};

Run runLinklet(const std::string &Src) {
  nora::DiagnosticEngine Diag;
  SourceStream S(Src.c_str(), &Diag);
  std::unique_ptr<ast::Linklet> AST = Parse::parseLinklet(S);
  REQUIRE(AST);
  Interpreter I(Diag);
  AST->accept(I);
  Run R;
  R.ok = !Diag.hadError();
  R.result = I.getResult();
  return R;
}

} // namespace

TEST_CASE("tail-recursive loop computes the correct value", "[interp][tco]") {
  Run R = runLinklet("(linklet () () "
                     "(letrec-values ([(loop) "
                     "  (lambda (n) (if (zero? n) 42 (loop (- n 1))))]) "
                     "(loop 1000)))");
  REQUIRE(R.ok);
  REQUIRE(R.result);
  auto *Int = llvm::dyn_cast<ast::Integer>(R.result.get());
  REQUIRE(Int);
  REQUIRE(*Int == 42);
}

namespace {
// Peak continuation depth of a self-tail-recursive countdown of `depth` steps.
size_t tailLoopPeak(int Depth) {
  nora::DiagnosticEngine Diag;
  std::string Src = "(linklet () () (letrec-values ([(loop) "
                    "(lambda (n) (if (zero? n) 0 (loop (- n 1))))]) (loop " +
                    std::to_string(Depth) + ")))";
  SourceStream S(Src.c_str(), &Diag);
  std::unique_ptr<ast::Linklet> AST = Parse::parseLinklet(S);
  REQUIRE(AST);
  Interpreter I(Diag);
  AST->accept(I);
  REQUIRE_FALSE(Diag.hadError());
  return I.getPeakKont();
}
} // namespace

TEST_CASE("tail recursion runs in bounded continuation space",
          "[interp][tco]") {
  // Proper tail calls: the same loop at wildly different iteration counts must
  // reach the *same* peak continuation depth (O(1)), and that depth is small.
  const size_t Shallow = tailLoopPeak(100);
  const size_t Deep = tailLoopPeak(100000);
  REQUIRE(Deep == Shallow);
  REQUIRE(Deep < 16);
}

namespace {
// Non-tail countdown: the recursive call sits under a pending (+ 1 ...), so it
// is not a tail call and must retain a continuation frame per level.
size_t nonTailLoopPeak(int Depth) {
  nora::DiagnosticEngine Diag;
  std::string Src = "(linklet () () (letrec-values ([(loop) "
                    "(lambda (n) (if (zero? n) 0 (+ 1 (loop (- n 1)))))]) "
                    "(loop " +
                    std::to_string(Depth) + ")))";
  SourceStream S(Src.c_str(), &Diag);
  std::unique_ptr<ast::Linklet> AST = Parse::parseLinklet(S);
  REQUIRE(AST);
  Interpreter I(Diag);
  AST->accept(I);
  REQUIRE_FALSE(Diag.hadError());
  return I.getPeakKont();
}
} // namespace

TEST_CASE("non-tail recursion still grows the continuation", "[interp][tco]") {
  // Contrast: proper tail calls must not collapse genuinely non-tail calls.
  // Peak depth grows with the recursion count and dwarfs the tail loop's.
  REQUIRE(nonTailLoopPeak(2000) > nonTailLoopPeak(1000));
  REQUIRE(nonTailLoopPeak(1000) > 10 * tailLoopPeak(1000));
}

TEST_CASE("a box round-trips its contents", "[interp][m2]") {
  Run R = runLinklet("(linklet () () (unbox (box 5)))");
  REQUIRE(R.ok);
  REQUIRE(R.result);
  auto *Int = llvm::dyn_cast<ast::Integer>(R.result.get());
  REQUIRE(Int);
  REQUIRE(*Int == 5);
}

TEST_CASE("set-box! mutates through a shared reference", "[interp][m2]") {
  // `b` is looked up three times (each lookup clones the value), yet the
  // mutation is visible: the box's cell is shared across the clones. This is
  // the behaviour the old clone-everything value model could not express.
  Run R = runLinklet("(linklet () () "
                     "(let-values ([(b) (box 1)]) "
                     "(begin (set-box! b 10) (unbox b))))");
  REQUIRE(R.ok);
  REQUIRE(R.result);
  auto *Int = llvm::dyn_cast<ast::Integer>(R.result.get());
  REQUIRE(Int);
  REQUIRE(*Int == 10);
}

TEST_CASE("eq? distinguishes box identity", "[interp][m2]") {
  // A box is eq? to itself; two freshly allocated boxes are not.
  Run Same =
      runLinklet("(linklet () () (let-values ([(b) (box 0)]) (eq? b b)))");
  REQUIRE(Same.ok);
  REQUIRE(Same.result);
  auto *S = llvm::dyn_cast<ast::BooleanLiteral>(Same.result.get());
  REQUIRE(S);
  REQUIRE(S->value());

  Run Diff = runLinklet("(linklet () () (eq? (box 0) (box 0)))");
  REQUIRE(Diff.ok);
  REQUIRE(Diff.result);
  auto *D = llvm::dyn_cast<ast::BooleanLiteral>(Diff.result.get());
  REQUIRE(D);
  REQUIRE_FALSE(D->value());
}

TEST_CASE("mutual tail recursion is bounded and correct", "[interp][tco]") {
  // ev/od tail-call each other: the reused activation frame belongs to a
  // *different* closure than the caller, so this exercises tail-call handling
  // in its general (non-self) form.
  Run R = runLinklet("(linklet () () (letrec-values ("
                     "  ((ev) (lambda (n) (if (zero? n) 1 (od (- n 1)))))"
                     "  ((od) (lambda (n) (if (zero? n) 0 (ev (- n 1))))))"
                     "  (ev 100000)))");
  REQUIRE(R.ok);
  REQUIRE(R.result);
  auto *Int = llvm::dyn_cast<ast::Integer>(R.result.get());
  REQUIRE(Int);
  REQUIRE(*Int == 1); // ev(100000): 100000 is even
}
