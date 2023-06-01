#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <llvm/Support/Casting.h>

#include "ast.h"

class Runtime {
public:
  // Disallow copying
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  std::unique_ptr<ast::ValueNode>
  callFunction(const std::string &Name,
               const std::vector<const ast::ValueNode *> &Args);

  bool isRuntimeFunction(const std::string &Name) {
    return RuntimeFunctions.find(Name) != RuntimeFunctions.end();
  }

  std::unique_ptr<ast::RuntimeFunction>
  lookupRuntimeFunction(const std::string &Name) {
    assert(isRuntimeFunction(Name) && "Function not found in runtime.");
    ast::ValueNode *V = RuntimeFunctions[Name]->clone();
    ast::RuntimeFunction *RF = llvm::cast<ast::RuntimeFunction>(V);
    return std::unique_ptr<ast::RuntimeFunction>(RF);
  }

  // Singleton instance
  static Runtime &getInstance() {
    static Runtime Instance;
    return Instance;
  }

private:
  Runtime(); /// Private constructor for singleton
  std::unordered_map<std::string, std::shared_ptr<ast::RuntimeFunction>>
      RuntimeFunctions;
};
