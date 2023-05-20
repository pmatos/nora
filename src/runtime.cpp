#include "runtime.h"

#include <llvm/Support/Casting.h>

class AddFunction : public ast::RuntimeFunction {
public:
  AddFunction(const std::string &Name) : RuntimeFunction(Name) {}

  virtual std::unique_ptr<ast::ValueNode>
  operator()(const std::vector<const ast::ValueNode *> &Args) const override {
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

  virtual std::unique_ptr<ast::ValueNode>
  operator()(const std::vector<const ast::ValueNode *> &Args) const override {
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

  virtual std::unique_ptr<ast::ValueNode>
  operator()(const std::vector<const ast::ValueNode *> &Args) const override {
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

#define RUNTIME_FUNC(Identifier, Name)                                         \
  RuntimeFunctions[Identifier] = std::make_shared<Name>(Identifier);
Runtime::Runtime() {
  // List of runtime functions.
  RUNTIME_FUNC("+", AddFunction);
  RUNTIME_FUNC("-", SubtractFunction);
  RUNTIME_FUNC("*", MultiplyFunction);
}

std::unique_ptr<ast::ValueNode>
Runtime::callFunction(const std::string &Name,
                      const std::vector<const ast::ValueNode *> &Args) {
  assert(RuntimeFunctions.find(Name) != RuntimeFunctions.end() &&
         "Function not found in runtime.");
  auto Fn = RuntimeFunctions[Name];
  return (*Fn)(Args);
}
