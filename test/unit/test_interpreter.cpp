#include <catch2/catch.hpp>

#include "AST.h"
#include "ASTRuntime.h"
#include "Diagnostics.h"
#include "Interpreter.h"
#include "Parse.h"
#include "SourceStream.h"

#include <llvm/Support/Casting.h>

#include <gc.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "nora_rt.h"

namespace {

// Parse + interpret a linklet source string at the interpreter's public seam.
struct Run {
  bool ok = false;                        // no diagnostics were reported
  std::unique_ptr<ast::ValueNode> result; // Interpreter::getResult()
  // Interpreter::getResultImmediate(): the M2/GC forcing seam, captured
  // alongside result regardless of getResult() being called first.
  std::optional<nr_value> RawImmediate;

  // The seam S18 will rewrite: downcast to a materialized ValueNode view.
  // Localized here so the eventual nr_value read replaces one definition.
  template <typename T>
  static const T *expectResult(const ast::ValueNode *Node) {
    REQUIRE(Node);
    const auto *Downcast = llvm::dyn_cast<T>(Node);
    REQUIRE(Downcast);
    return Downcast;
  }

  static void expectInt(const ast::ValueNode *Node, int64_t Expected) {
    REQUIRE(*expectResult<ast::Integer>(Node) == Expected);
  }

  void expectInt(int64_t Expected) const {
    REQUIRE(ok);
    expectInt(result.get(), Expected);
  }

  static void expectBool(const ast::ValueNode *Node, bool Expected) {
    REQUIRE(expectResult<ast::BooleanLiteral>(Node)->value() == Expected);
  }

  void expectBool(bool Expected) const {
    REQUIRE(ok);
    expectBool(result.get(), Expected);
  }

  static void expectChar(const ast::ValueNode *Node, uint32_t Expected) {
    REQUIRE(expectResult<ast::Char>(Node)->getCodePoint() == Expected);
  }

  void expectChar(uint32_t Expected) const {
    REQUIRE(ok);
    expectChar(result.get(), Expected);
  }

  static void expectVoid(const ast::ValueNode *Node) {
    expectResult<ast::Void>(Node);
  }

  void expectVoid() const {
    REQUIRE(ok);
    expectVoid(result.get());
  }
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
  R.RawImmediate = I.getResultImmediate();
  R.result = I.getResult();
  return R;
}

} // namespace

TEST_CASE("tail-recursive loop computes the correct value", "[interp][tco]") {
  Run R = runLinklet("(linklet () () "
                     "(letrec-values ([(loop) "
                     "  (lambda (n) (if (zero? n) 42 (loop (- n 1))))]) "
                     "(loop 1000)))");
  R.expectInt(42);
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

TEST_CASE("the continuation lives in the GC heap", "[m2][gc]") {
  // A deep non-tail recursion grows the Kont vector to thousands of frames.
  // Nothing else is GC-allocated during evaluation yet, so the cumulative GC
  // bytes churned by the run measure Kont's backing store: ~0 while Kont is
  // malloc'd, large once it is GC-allocated. (GC-heap seam, per the plan.)
  const size_t Before = GC_get_total_bytes();
  (void)nonTailLoopPeak(4000);
  const size_t Churned = GC_get_total_bytes() - Before;
  REQUIRE(Churned > 100000);
}

TEST_CASE("a box round-trips its contents", "[interp][m2]") {
  Run R = runLinklet("(linklet () () (unbox (box 5)))");
  R.expectInt(5);
}

TEST_CASE("set-box! mutates through a shared reference", "[interp][m2]") {
  // `b` is looked up three times (each lookup clones the value), yet the
  // mutation is visible: the box's cell is shared across the clones. This is
  // the behaviour the old clone-everything value model could not express.
  Run R = runLinklet("(linklet () () "
                     "(let-values ([(b) (box 1)]) "
                     "(begin (set-box! b 10) (unbox b))))");
  R.expectInt(10);
}

TEST_CASE("eq? distinguishes box identity", "[interp][m2]") {
  // A box is eq? to itself; two freshly allocated boxes are not.
  Run Same =
      runLinklet("(linklet () () (let-values ([(b) (box 0)]) (eq? b b)))");
  Same.expectBool(true);

  Run Diff = runLinklet("(linklet () () (eq? (box 0) (box 0)))");
  Diff.expectBool(false);
}

TEST_CASE("cons/car/cdr round-trip", "[interp][m2]") {
  Run Ca = runLinklet("(linklet () () (car (cons 1 2)))");
  Ca.expectInt(1);

  Run Cd = runLinklet("(linklet () () (cdr (cons 1 2)))");
  Cd.expectInt(2);
}

TEST_CASE("set-car!/set-cdr! mutate through a shared reference",
          "[interp][m2]") {
  Run R = runLinklet(
      "(linklet () () (let-values ([(p) (cons 1 2)]) "
      "(begin (set-car! p 10) (set-cdr! p 20) (+ (car p) (cdr p)))))");
  R.expectInt(30);
}

TEST_CASE("eq? distinguishes pair identity", "[interp][m2]") {
  Run Same =
      runLinklet("(linklet () () (let-values ([(p) (cons 1 2)]) (eq? p p)))");
  Same.expectBool(true);

  Run Diff = runLinklet("(linklet () () (eq? (cons 1 2) (cons 1 2)))");
  Diff.expectBool(false);
}

TEST_CASE("symbol eq? is identity, not name", "[interp][m2]") {
  // Two uninterned symbols with the same name are distinct objects...
  Run Un = runLinklet("(linklet () () (eq? (string->uninterned-symbol \"s\") "
                      "(string->uninterned-symbol \"s\")))");
  Un.expectBool(false);

  // ...while interned symbols with the same name are eq?.
  Run In = runLinklet("(linklet () () (eq? 'a 'a))");
  In.expectBool(true);
}

TEST_CASE("eq? unwraps a quoted symbol before comparing identity",
          "[interp][m2]") {
  // A quoted symbol literal like 'probe evaluates to a QuotedExpr wrapping
  // the interned symbol, not a bare Symbol; eq? must unwrap it before its
  // identity dispatch rather than falling through to valueEq's structural
  // (name-only) comparison, which would wrongly equate it with an
  // uninterned symbol of the same name. Built directly against the Runtime
  // seam (rather than parsed source) so the comparison has same-named
  // operands deterministically, independent of gensym's shared counter.
  auto Quoted = std::make_unique<ast::QuotedExpr>();
  Quoted->setQuotedExpr(std::make_unique<ast::Symbol>("probe"));
  std::unique_ptr<ast::Symbol> Uninterned =
      ast::Symbol::makeUninterned("probe");

  llvm::SmallVector<const ast::ValueNode *> Args = {Quoted.get(),
                                                    Uninterned.get()};
  std::unique_ptr<ast::ValueNode> Result =
      Runtime::getInstance().callFunction("eq?", Args);
  Run::expectBool(Result.get(), false);

  // Same check with operands swapped.
  llvm::SmallVector<const ast::ValueNode *> ArgsRev = {Uninterned.get(),
                                                       Quoted.get()};
  std::unique_ptr<ast::ValueNode> ResultRev =
      Runtime::getInstance().callFunction("eq?", ArgsRev);
  Run::expectBool(ResultRev.get(), false);
}

TEST_CASE("gensym produces fresh distinct symbols", "[interp][m2]") {
  Run R = runLinklet("(linklet () () (eq? (gensym) (gensym)))");
  R.expectBool(false);
}

TEST_CASE("gensym rejects more than one argument", "[interp][m2]") {
  Run R = runLinklet("(linklet () () (gensym 'a 'b))");
  REQUIRE_FALSE(R.ok);
}

TEST_CASE("mutual tail recursion is bounded and correct", "[interp][tco]") {
  // ev/od tail-call each other: the reused activation frame belongs to a
  // *different* closure than the caller, so this exercises tail-call handling
  // in its general (non-self) form.
  Run R = runLinklet("(linklet () () (letrec-values ("
                     "  ((ev) (lambda (n) (if (zero? n) 1 (od (- n 1)))))"
                     "  ((od) (lambda (n) (if (zero? n) 0 (ev (- n 1))))))"
                     "  (ev 100000)))");
  R.expectInt(1); // ev(100000): 100000 is even
}

namespace {
// Peak continuation depth of a self-tail-recursive countdown of `depth` steps
// whose body is wrapped in a with-continuation-mark around the tail call.
size_t wcmTailLoopPeak(int Depth) {
  nora::DiagnosticEngine Diag;
  std::string Src = "(linklet () () (letrec-values ([(loop) "
                    "(lambda (n) (with-continuation-mark 'k n "
                    "  (if (zero? n) 0 (loop (- n 1)))))]) (loop " +
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

TEST_CASE("a tail call through with-continuation-mark runs in bounded space",
          "[interp][tco][m2]") {
  // A with-continuation-mark wrapping a self-tail-recursive loop's body must
  // not defeat frame reuse: the WcmMark frame it installs is still on top of
  // Kont when the tail call happens, so this must reach the same *small*,
  // depth-independent peak as an unwrapped tail loop.
  const size_t Shallow = wcmTailLoopPeak(100);
  const size_t Deep = wcmTailLoopPeak(100000);
  REQUIRE(Deep == Shallow);
  REQUIRE(Deep < 16);
}

TEST_CASE("a tail-position with-continuation-mark replaces, not "
          "accumulates, a same-key mark across loop iterations",
          "[interp][m2]") {
  // Each iteration's with-continuation-mark is in tail position, so
  // successive iterations share one continuation frame; installing the same
  // key there again must replace the previous value (as real Scheme/Racket
  // does), not stack a second entry alongside it.
  Run R = runLinklet("(linklet () () (letrec-values ([(loop) "
                     "(lambda (n) (with-continuation-mark 'k n "
                     "  (if (zero? n) (continuation-mark-set->list "
                     "                  (current-continuation-marks) 'k) "
                     "      (loop (- n 1)))))]) (loop 5)))");
  REQUIRE(R.ok);
  auto *L = Run::expectResult<ast::List>(R.result.get());
  REQUIRE(L->length() == 1);
  Run::expectInt(&(*L)[0], 0);
}

TEST_CASE("a box installed as a continuation mark keeps its identity",
          "[interp][m2]") {
  // Nothing else exercises identity *through* a continuation mark: a Box's
  // own shared cell already guarantees eq?/mutation survive a clone
  // regardless of the mark storage's own type, so this passes both before
  // and after MarkFrame/WcmKeyV move to Value - it pins the invariant going
  // forward rather than proving new behaviour.
  Run R = runLinklet("(linklet () () "
                     "(let-values ([(b) (box 1)]) "
                     "  (with-continuation-mark 'k b "
                     "    (begin "
                     "      (set-box! (continuation-mark-set-first "
                     "                  (current-continuation-marks) 'k) 42) "
                     "      (unbox b)))))");
  R.expectInt(42);
}

TEST_CASE("#f literal result is the NR_FALSE immediate, not an allocated "
          "BooleanLiteral",
          "[interp][m2][gc]") {
  Run R = runLinklet("(linklet () () #f)");
  REQUIRE(R.ok);
  REQUIRE(R.RawImmediate.has_value());
  REQUIRE(*R.RawImmediate == NR_FALSE);
  R.expectBool(false);
}

TEST_CASE("if branches on a let-bound #f via the materialized fallback, not "
          "the immediate fast path",
          "[interp][m2][gc]") {
  // x is bound via Environment/toShared(), which materializes the immediate
  // into a real ast::BooleanLiteral - this pins step(IfBranch)'s
  // dyn_cast_or_null<BooleanLiteral> fallback so it isn't deleted alongside
  // the new nr_truthy fast path.
  Run R = runLinklet("(linklet () () (let-values ([(x) #f]) (if x 1 2)))");
  R.expectInt(2);
}

TEST_CASE("eq? on immediate-boolean arguments materializes at the "
          "RuntimeFunction boundary",
          "[interp][m2][gc]") {
  // Nothing before S6 ever passed a bare boolean literal to a
  // RuntimeFunction; EqFunction dereferences its Args unconditionally, so
  // this would segfault if Value::get() returned null for an engaged
  // immediate instead of materializing it.
  Run Same = runLinklet("(linklet () () (eq? #t #t))");
  Same.expectBool(true);

  Run Diff = runLinklet("(linklet () () (eq? #t #f))");
  Diff.expectBool(false);
}

TEST_CASE("a box can hold and return an immediate boolean",
          "[interp][m2][gc]") {
  Run R = runLinklet("(linklet () () (unbox (box #t)))");
  R.expectBool(true);
}

TEST_CASE("bare char literal result is the nr_char immediate, not an "
          "allocated Char",
          "[interp][m2][gc]") {
  Run R = runLinklet("(linklet () () #\\a)");
  REQUIRE(R.ok);
  REQUIRE(R.RawImmediate.has_value());
  REQUIRE(*R.RawImmediate == nr_char('a'));
  R.expectChar('a');
}

TEST_CASE("define-values result is the NR_VOID immediate, not an allocated "
          "Void",
          "[interp][m2][gc]") {
  Run R = runLinklet("(linklet () () (define-values (x) 5))");
  REQUIRE(R.ok);
  REQUIRE(R.RawImmediate.has_value());
  REQUIRE(*R.RawImmediate == NR_VOID);
  R.expectVoid();
}

TEST_CASE("set! result is void", "[interp][m2][gc]") {
  // Whether this also round-trips through RawImmediate depends on whether the
  // enclosing let-values body frame materializes the result on its way out
  // (the same Call/WcmMark deferral S6 already documented for booleans) - the
  // forcing case above (top-level define-values) is what pins RawImmediate.
  Run R = runLinklet("(linklet () () (let-values ([(x) 1]) (set! x 2)))");
  R.expectVoid();
}
