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
}

std::unique_ptr<ast::ValueNode>
Runtime::callFunction(const std::string &Name,
                      const llvm::SmallVector<const ast::ValueNode *> &Args) {
  assert(RuntimeFunctions.find(Name) != RuntimeFunctions.end() &&
         "Function not found in runtime.");
  auto Fn = RuntimeFunctions[Name];
  return (*Fn)(Args);
}
