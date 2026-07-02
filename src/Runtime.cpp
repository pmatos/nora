#include "Runtime.h"

#include "ASTRuntime.h"

#include <llvm/Support/Casting.h>

class AddFunction : public ast::RuntimeFunction {
public:
  AddFunction(const std::string &Name) : RuntimeFunction(Name) {}

  virtual std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    auto Sum = std::make_unique<ast::Integer>(0);

    for (const auto *Arg : Args) {
      if (auto const *I = llvm::dyn_cast<ast::Integer>(Arg)) {
        *Sum += *I;
      } else {
        // FIXME: Throw an error. Unsupported type for +.
        return nullptr;
      }
    }
    return Sum;
  }

  virtual ast::RuntimeFunction *clone() const override {
    return new AddFunction(*this);
  }

  virtual void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class SubtractFunction : public ast::RuntimeFunction {
public:
  SubtractFunction(const std::string &Name) : RuntimeFunction(Name) {}

  virtual std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.empty()) {
      // Runtime error, no args to -
      // FIXME: issue error message
      return nullptr;
    }

    std::unique_ptr<ast::Integer> Sub;
    bool First = true;

    for (const auto *Arg : Args) {
      if (auto const *I = llvm::dyn_cast<ast::Integer>(Arg)) {
        if (First) {

          Sub = std::unique_ptr<ast::Integer>(
              llvm::cast<ast::Integer>(I->clone()));
          First = false;
        } else {
          *Sub -= *I;
        }
      } else {
        // FIXME: Throw an error. Unsupported type for -.
        return nullptr;
      }
    }

    return Sub;
  }

  virtual ast::RuntimeFunction *clone() const override {
    return new SubtractFunction(*this);
  }

  virtual void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class MultiplyFunction : public ast::RuntimeFunction {
public:
  MultiplyFunction(const std::string &Name) : RuntimeFunction(Name) {}

  virtual std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    auto Mul = std::make_unique<ast::Integer>(1);

    for (const auto *Arg : Args) {
      if (auto const *I = llvm::dyn_cast<ast::Integer>(Arg)) {
        *Mul *= *I;
      } else {
        // FIXME: Throw an error. Unsupported type for -.
        return nullptr;
      }
    }

    return Mul;
  }

  virtual ast::RuntimeFunction *clone() const override {
    return new MultiplyFunction(*this);
  }

  virtual void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (current-continuation-marks) is intercepted by the interpreter, which has
// access to the machine's continuation. This runtime entry exists only so the
// identifier resolves to a callable value; it is not invoked in practice.
class CurrentContinuationMarksFunction : public ast::RuntimeFunction {
public:
  CurrentContinuationMarksFunction(const std::string &Name)
      : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    (void)Args;
    return std::make_unique<ast::ContinuationMarkSet>();
  }

  ast::RuntimeFunction *clone() const override {
    return new CurrentContinuationMarksFunction(*this);
  }

  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (continuation-mark-set-first mark-set key) returns the innermost (most
// recent) value marked with key, or #f if there is none.
class ContinuationMarkSetFirstFunction : public ast::RuntimeFunction {
public:
  ContinuationMarkSetFirstFunction(const std::string &Name)
      : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() == 2) {
      const auto *CMS = llvm::dyn_cast<ast::ContinuationMarkSet>(Args[0]);
      const ast::ValueNode *Key = Args[1];
      if (CMS != nullptr && Key != nullptr) {
        for (auto const &Frame : CMS->getFrames()) {
          for (auto const &E : Frame) {
            if (ast::valueEq(*E.first, *Key)) {
              return std::unique_ptr<ast::ValueNode>(E.second->clone());
            }
          }
        }
      }
    }
    return std::make_unique<ast::BooleanLiteral>(false);
  }

  ast::RuntimeFunction *clone() const override {
    return new ContinuationMarkSetFirstFunction(*this);
  }

  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (continuation-mark-set->list mark-set key) returns the list of values marked
// with key, ordered from innermost (most recent) to outermost.
class ContinuationMarkSetToListFunction : public ast::RuntimeFunction {
public:
  ContinuationMarkSetToListFunction(const std::string &Name)
      : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    auto L = std::make_unique<ast::List>();
    if (Args.size() == 2) {
      const auto *CMS = llvm::dyn_cast<ast::ContinuationMarkSet>(Args[0]);
      const ast::ValueNode *Key = Args[1];
      if (CMS != nullptr && Key != nullptr) {
        for (auto const &Frame : CMS->getFrames()) {
          for (auto const &E : Frame) {
            if (ast::valueEq(*E.first, *Key)) {
              L->appendExpr(std::unique_ptr<ast::ValueNode>(E.second->clone()));
            }
          }
        }
      }
    }
    return L;
  }

  ast::RuntimeFunction *clone() const override {
    return new ContinuationMarkSetToListFunction(*this);
  }

  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (zero? n) is a minimal integer predicate. It exists so a terminating
// tail-recursive loop can be written to exercise proper tail calls (M1); the
// full numeric tower and its predicates arrive in M4.
class ZeroPredicateFunction : public ast::RuntimeFunction {
public:
  ZeroPredicateFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 1) {
      return nullptr;
    }
    if (auto const *I = llvm::dyn_cast<ast::Integer>(Args[0])) {
      return std::make_unique<ast::BooleanLiteral>(*I == 0);
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new ZeroPredicateFunction(*this);
  }

  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (box v) allocates a fresh mutable cell holding v. (unbox b) reads it. The
// box's cell is shared across copies of the Box value, so mutation and identity
// survive the interpreter's clone-on-lookup - the start of M2's shared value
// model.
class BoxFunction : public ast::RuntimeFunction {
public:
  BoxFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 1) {
      return nullptr;
    }
    return std::make_unique<ast::Box>(
        std::unique_ptr<ast::ValueNode>(Args[0]->clone()));
  }

  ast::RuntimeFunction *clone() const override {
    return new BoxFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class UnboxFunction : public ast::RuntimeFunction {
public:
  UnboxFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 1) {
      return nullptr;
    }
    if (auto const *B = llvm::dyn_cast<ast::Box>(Args[0])) {
      return B->get();
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new UnboxFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class SetBoxFunction : public ast::RuntimeFunction {
public:
  SetBoxFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 2) {
      return nullptr;
    }
    if (auto const *B = llvm::dyn_cast<ast::Box>(Args[0])) {
      B->set(std::unique_ptr<ast::ValueNode>(Args[1]->clone()));
      return std::make_unique<ast::Void>();
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new SetBoxFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (eq? a b): object identity. Heap objects with a cell (boxes) compare by cell
// pointer; other values fall back to the structural valueEq approximation
// (interned symbols, fixnums, chars, booleans). This is the identity operation
// the clone-everything model could not provide.
class EqFunction : public ast::RuntimeFunction {
public:
  EqFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 2) {
      return nullptr;
    }
    const ast::ValueNode *A = Args[0];
    const ast::ValueNode *B = Args[1];
    bool Eq;
    if (auto const *BA = llvm::dyn_cast<ast::Box>(A)) {
      auto const *BB = llvm::dyn_cast<ast::Box>(B);
      Eq = (BB != nullptr) && BA->identity() == BB->identity();
    } else if (auto const *PA = llvm::dyn_cast<ast::Pair>(A)) {
      auto const *PB = llvm::dyn_cast<ast::Pair>(B);
      Eq = (PB != nullptr) && PA->identity() == PB->identity();
    } else {
      Eq = ast::valueEq(*A, *B);
    }
    return std::make_unique<ast::BooleanLiteral>(Eq);
  }

  ast::RuntimeFunction *clone() const override { return new EqFunction(*this); }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

// (cons a d) allocates a fresh mutable pair; (car p)/(cdr p) read its fields.
// The pair's cell is shared across copies of the Pair value, so mutation and
// identity survive the interpreter's clone-on-lookup.
class ConsFunction : public ast::RuntimeFunction {
public:
  ConsFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 2) {
      return nullptr;
    }
    return std::make_unique<ast::Pair>(
        std::unique_ptr<ast::ValueNode>(Args[0]->clone()),
        std::unique_ptr<ast::ValueNode>(Args[1]->clone()));
  }

  ast::RuntimeFunction *clone() const override {
    return new ConsFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class CarFunction : public ast::RuntimeFunction {
public:
  CarFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 1) {
      return nullptr;
    }
    if (auto const *P = llvm::dyn_cast<ast::Pair>(Args[0])) {
      return P->car();
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new CarFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class CdrFunction : public ast::RuntimeFunction {
public:
  CdrFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 1) {
      return nullptr;
    }
    if (auto const *P = llvm::dyn_cast<ast::Pair>(Args[0])) {
      return P->cdr();
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new CdrFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class SetCarFunction : public ast::RuntimeFunction {
public:
  SetCarFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 2) {
      return nullptr;
    }
    if (auto const *P = llvm::dyn_cast<ast::Pair>(Args[0])) {
      P->setCar(std::unique_ptr<ast::ValueNode>(Args[1]->clone()));
      return std::make_unique<ast::Void>();
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new SetCarFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

class SetCdrFunction : public ast::RuntimeFunction {
public:
  SetCdrFunction(const std::string &Name) : RuntimeFunction(Name) {}

  std::unique_ptr<ast::ValueNode> operator()(
      const llvm::SmallVector<const ast::ValueNode *> &Args) const override {
    if (Args.size() != 2) {
      return nullptr;
    }
    if (auto const *P = llvm::dyn_cast<ast::Pair>(Args[0])) {
      P->setCdr(std::unique_ptr<ast::ValueNode>(Args[1]->clone()));
      return std::make_unique<ast::Void>();
    }
    return nullptr;
  }

  ast::RuntimeFunction *clone() const override {
    return new SetCdrFunction(*this);
  }
  void accept(ASTVisitor &V) const override { V.visit(*this); }
};

#define RUNTIME_FUNC(Identifier, Name)                                         \
  RuntimeFunctions[Identifier] = std::make_shared<Name>(Identifier);
Runtime::Runtime() {
  // List of runtime functions.
  RUNTIME_FUNC("+", AddFunction);
  RUNTIME_FUNC("-", SubtractFunction);
  RUNTIME_FUNC("*", MultiplyFunction);
  RUNTIME_FUNC("current-continuation-marks", CurrentContinuationMarksFunction);
  RUNTIME_FUNC("continuation-mark-set-first", ContinuationMarkSetFirstFunction);
  RUNTIME_FUNC("continuation-mark-set->list",
               ContinuationMarkSetToListFunction);
  RUNTIME_FUNC("zero?", ZeroPredicateFunction);
  RUNTIME_FUNC("box", BoxFunction);
  RUNTIME_FUNC("unbox", UnboxFunction);
  RUNTIME_FUNC("set-box!", SetBoxFunction);
  RUNTIME_FUNC("eq?", EqFunction);
  RUNTIME_FUNC("cons", ConsFunction);
  RUNTIME_FUNC("car", CarFunction);
  RUNTIME_FUNC("cdr", CdrFunction);
  RUNTIME_FUNC("set-car!", SetCarFunction);
  RUNTIME_FUNC("set-cdr!", SetCdrFunction);
}

std::unique_ptr<ast::ValueNode>
Runtime::callFunction(const std::string &Name,
                      const llvm::SmallVector<const ast::ValueNode *> &Args) {
  assert(RuntimeFunctions.find(Name) != RuntimeFunctions.end() &&
         "Function not found in runtime.");
  auto Fn = RuntimeFunctions[Name];
  return (*Fn)(Args);
}
