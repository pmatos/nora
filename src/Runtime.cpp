#include "Runtime.h"

#include "AST.h"
#include "ASTRuntime.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include <memory>
#include <string>

using Args = const llvm::SmallVector<const ast::ValueNode *> &;

// Every builtin is registered as data: a name, an arity spec the seam checks
// before dispatch, and a handler carrying only its core logic. The arity guard,
// the clone/accept plumbing, and the node kind that each hand-rolled subclass
// used to repeat now live once, behind this seam.
Runtime::Runtime() {
  Builtins["+"] = {Arity::any(), [](Args A) -> std::unique_ptr<ast::ValueNode> {
                     auto Sum = std::make_unique<ast::Integer>(0);
                     for (const auto *Arg : A) {
                       if (auto const *I = llvm::dyn_cast<ast::Integer>(Arg)) {
                         *Sum += *I;
                       } else {
                         return nullptr;
                       }
                     }
                     return Sum;
                   }};

  Builtins["-"] = {Arity::atLeast(1),
                   [](Args A) -> std::unique_ptr<ast::ValueNode> {
                     std::unique_ptr<ast::Integer> Sub;
                     bool First = true;
                     for (const auto *Arg : A) {
                       if (auto const *I = llvm::dyn_cast<ast::Integer>(Arg)) {
                         if (First) {
                           Sub = std::unique_ptr<ast::Integer>(
                               llvm::cast<ast::Integer>(I->clone()));
                           First = false;
                         } else {
                           *Sub -= *I;
                         }
                       } else {
                         return nullptr;
                       }
                     }
                     return Sub;
                   }};

  Builtins["*"] = {Arity::any(), [](Args A) -> std::unique_ptr<ast::ValueNode> {
                     auto Mul = std::make_unique<ast::Integer>(1);
                     for (const auto *Arg : A) {
                       if (auto const *I = llvm::dyn_cast<ast::Integer>(Arg)) {
                         *Mul *= *I;
                       } else {
                         return nullptr;
                       }
                     }
                     return Mul;
                   }};

  // (current-continuation-marks) is intercepted by the interpreter, which has
  // access to the machine's continuation. This entry exists only so the
  // identifier resolves to a callable value; it is not invoked in practice.
  Builtins["current-continuation-marks"] = {
      Arity::any(), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        (void)A;
        return std::make_unique<ast::ContinuationMarkSet>();
      }};

  // (continuation-mark-set-first mark-set key) returns the innermost value
  // marked with key, or #f if there is none. Note: a wrong-arity call yields #f
  // rather than the nullptr error channel, so the size guard stays inline and
  // the arity is `any`.
  Builtins["continuation-mark-set-first"] = {
      Arity::any(), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        if (A.size() == 2) {
          const auto *CMS = llvm::dyn_cast<ast::ContinuationMarkSet>(A[0]);
          const ast::ValueNode *Key = A[1];
          if (CMS != nullptr && Key != nullptr) {
            for (auto const &Frame : CMS->getFrames()) {
              for (auto const &E : Frame) {
                if (ast::valueEq(*E.first.get(), *Key)) {
                  return std::unique_ptr<ast::ValueNode>(
                      E.second.get()->clone());
                }
              }
            }
          }
        }
        return std::make_unique<ast::BooleanLiteral>(false);
      }};

  // (continuation-mark-set->list mark-set key) returns the values marked with
  // key, innermost first. A wrong-arity call yields the empty list, not an
  // error, so the size guard stays inline and the arity is `any`.
  Builtins["continuation-mark-set->list"] = {
      Arity::any(), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        auto L = std::make_unique<ast::List>();
        if (A.size() == 2) {
          const auto *CMS = llvm::dyn_cast<ast::ContinuationMarkSet>(A[0]);
          const ast::ValueNode *Key = A[1];
          if (CMS != nullptr && Key != nullptr) {
            for (auto const &Frame : CMS->getFrames()) {
              for (auto const &E : Frame) {
                if (ast::valueEq(*E.first.get(), *Key)) {
                  L->appendExpr(
                      std::unique_ptr<ast::ValueNode>(E.second.get()->clone()));
                }
              }
            }
          }
        }
        return L;
      }};

  // (zero? n) is a minimal integer predicate (M1's tail-call harness); the full
  // numeric tower arrives in M4.
  Builtins["zero?"] = {
      Arity::exactly(1), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        if (auto const *I = llvm::dyn_cast<ast::Integer>(A[0])) {
          return std::make_unique<ast::BooleanLiteral>(*I == 0);
        }
        return nullptr;
      }};

  // (box v)/(unbox b)/(set-box! b v): a fresh mutable cell shared across copies
  // of the Box value, so mutation and identity survive whenever a clone is
  // materialized from a shared environment binding.
  Builtins["box"] = {Arity::exactly(1),
                     [](Args A) -> std::unique_ptr<ast::ValueNode> {
                       return std::make_unique<ast::Box>(
                           std::unique_ptr<ast::ValueNode>(A[0]->clone()));
                     }};

  Builtins["unbox"] = {Arity::exactly(1),
                       [](Args A) -> std::unique_ptr<ast::ValueNode> {
                         if (auto const *B = llvm::dyn_cast<ast::Box>(A[0])) {
                           return B->get();
                         }
                         return nullptr;
                       }};

  Builtins["set-box!"] = {
      Arity::exactly(2), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        if (auto const *B = llvm::dyn_cast<ast::Box>(A[0])) {
          B->set(std::unique_ptr<ast::ValueNode>(A[1]->clone()));
          return std::make_unique<ast::Void>();
        }
        return nullptr;
      }};

  // (eq? a b): object identity. Heap objects with a cell (boxes, pairs) compare
  // by cell pointer; symbols by interned identity; other values fall back to
  // the structural valueEq approximation. A quoted datum is unwrapped first so
  // 'k does not skip the identity branches.
  Builtins["eq?"] = {
      Arity::exactly(2), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        const ast::ValueNode *AA = A[0];
        const ast::ValueNode *B = A[1];
        while (auto const *QA = llvm::dyn_cast<ast::QuotedExpr>(AA)) {
          AA = &QA->getQuotedExpr();
        }
        while (auto const *QB = llvm::dyn_cast<ast::QuotedExpr>(B)) {
          B = &QB->getQuotedExpr();
        }
        bool Eq;
        if (auto const *BA = llvm::dyn_cast<ast::Box>(AA)) {
          auto const *BB = llvm::dyn_cast<ast::Box>(B);
          Eq = (BB != nullptr) && BA->identity() == BB->identity();
        } else if (auto const *PA = llvm::dyn_cast<ast::Pair>(AA)) {
          auto const *PB = llvm::dyn_cast<ast::Pair>(B);
          Eq = (PB != nullptr) && PA->identity() == PB->identity();
        } else if (auto const *SA = llvm::dyn_cast<ast::Symbol>(AA)) {
          auto const *SB = llvm::dyn_cast<ast::Symbol>(B);
          Eq = (SB != nullptr) && SA->identity() == SB->identity();
        } else {
          Eq = ast::valueEq(*AA, *B);
        }
        return std::make_unique<ast::BooleanLiteral>(Eq);
      }};

  // (cons a d)/(car p)/(cdr p)/(set-car! p v)/(set-cdr! p v): a fresh mutable
  // pair whose cell is shared across copies of the Pair value.
  Builtins["cons"] = {Arity::exactly(2),
                      [](Args A) -> std::unique_ptr<ast::ValueNode> {
                        return std::make_unique<ast::Pair>(
                            std::unique_ptr<ast::ValueNode>(A[0]->clone()),
                            std::unique_ptr<ast::ValueNode>(A[1]->clone()));
                      }};

  Builtins["car"] = {Arity::exactly(1),
                     [](Args A) -> std::unique_ptr<ast::ValueNode> {
                       if (auto const *P = llvm::dyn_cast<ast::Pair>(A[0])) {
                         return P->car();
                       }
                       return nullptr;
                     }};

  Builtins["cdr"] = {Arity::exactly(1),
                     [](Args A) -> std::unique_ptr<ast::ValueNode> {
                       if (auto const *P = llvm::dyn_cast<ast::Pair>(A[0])) {
                         return P->cdr();
                       }
                       return nullptr;
                     }};

  Builtins["set-car!"] = {
      Arity::exactly(2), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        if (auto const *P = llvm::dyn_cast<ast::Pair>(A[0])) {
          P->setCar(std::unique_ptr<ast::ValueNode>(A[1]->clone()));
          return std::make_unique<ast::Void>();
        }
        return nullptr;
      }};

  Builtins["set-cdr!"] = {
      Arity::exactly(2), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        if (auto const *P = llvm::dyn_cast<ast::Pair>(A[0])) {
          P->setCdr(std::unique_ptr<ast::ValueNode>(A[1]->clone()));
          return std::make_unique<ast::Void>();
        }
        return nullptr;
      }};

  // (string->uninterned-symbol s): a fresh uninterned symbol, distinct from
  // every other symbol even one with the same name.
  Builtins["string->uninterned-symbol"] = {
      Arity::exactly(1), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        if (auto const *S = llvm::dyn_cast<ast::String>(A[0])) {
          return ast::Symbol::makeUninterned(S->getValue());
        }
        return nullptr;
      }};

  // (gensym [base]) returns a fresh uninterned symbol, never eq? to any other.
  // A monotonic counter gives it a readable, unique name; distinctness comes
  // from its uninterned identity, not the name. A wrong-typed base is ignored.
  Builtins["gensym"] = {
      Arity::atMost(1), [](Args A) -> std::unique_ptr<ast::ValueNode> {
        static unsigned Counter = 0;
        std::string Base = "g";
        if (A.size() == 1) {
          if (auto const *S = llvm::dyn_cast<ast::Symbol>(A[0])) {
            Base = S->getName().str();
          } else if (auto const *Str = llvm::dyn_cast<ast::String>(A[0])) {
            Base = Str->getValue().str();
          }
        }
        return ast::Symbol::makeUninterned(Base + std::to_string(++Counter));
      }};
}

std::unique_ptr<ast::ValueNode>
Runtime::callFunction(const std::string &Name,
                      const llvm::SmallVector<const ast::ValueNode *> &Args) {
  auto It = Builtins.find(Name);
  if (It == Builtins.end() || !It->second.Ar.accepts(Args.size())) {
    return nullptr;
  }
  return It->second.Fn(Args);
}
