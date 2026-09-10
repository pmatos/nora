#pragma once

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <llvm/ADT/SmallVector.h>

#include "AST.h"

class Runtime {
public:
  // How many arguments a builtin accepts: the inclusive range [Min, Max], with
  // Unbounded as the upper end for a variadic builtin. This is the single arity
  // spec the seam validates before dispatching, replacing the per-builtin
  // `if (Args.size() != N) return nullptr;` prologues.
  struct Arity {
    static constexpr std::size_t Unbounded = static_cast<std::size_t>(-1);
    std::size_t Min;
    std::size_t Max;
    static Arity exactly(std::size_t N) { return {N, N}; }
    static Arity atLeast(std::size_t N) { return {N, Unbounded}; }
    static Arity atMost(std::size_t N) { return {0, N}; }
    static Arity any() { return {0, Unbounded}; }
    bool accepts(std::size_t N) const { return N >= Min && N <= Max; }
  };

  // A builtin's behaviour, invoked with arguments the seam has already
  // arity-checked. Returns nullptr to signal a type error (the caller renders
  // it); a builtin whose failure semantics differ keeps that logic inline.
  using Handler = std::function<std::unique_ptr<ast::ValueNode>(
      const llvm::SmallVector<const ast::ValueNode *> &)>;

  // Disallow copying
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  std::unique_ptr<ast::ValueNode>
  callFunction(const std::string &Name,
               const llvm::SmallVector<const ast::ValueNode *> &Args);

  bool isRuntimeFunction(const std::string &Name) {
    return Builtins.find(Name) != Builtins.end();
  }

  std::unique_ptr<ast::RuntimeFunction>
  lookupRuntimeFunction(const std::string &Name) {
    assert(isRuntimeFunction(Name) && "Function not found in runtime.");
    return std::make_unique<ast::RuntimeFunction>(Name);
  }

  // Singleton instance
  static Runtime &getInstance() {
    static Runtime Instance;
    return Instance;
  }

private:
  Runtime(); /// Private constructor for singleton
  struct Builtin {
    Arity Ar;
    Handler Fn;
  };
  std::unordered_map<std::string, Builtin> Builtins;
};
