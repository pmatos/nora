#include "runtime.h"

#include <llvm/Support/Casting.h>

class AddFunction : public ast::RuntimeFunction {
public:
  AddFunction(const std::wstring &Name) : RuntimeFunction(Name) {}

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

#define RUNTIME_FUNC(Identifier, Name)                                         \
  RuntimeFunctions[Identifier] = std::make_shared<Name>(Identifier);
Runtime::Runtime() {
  // List of runtime functions.
  RUNTIME_FUNC(L"+", AddFunction);
}

std::unique_ptr<ast::ValueNode>
Runtime::callFunction(const std::wstring &Name,
                      const std::vector<const ast::ValueNode *> &Args) {
  assert(RuntimeFunctions.find(Name) != RuntimeFunctions.end() &&
         "Function not found in runtime.");
  auto Fn = RuntimeFunctions[Name];
  return (*Fn)(Args);
}
