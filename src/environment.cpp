#include "environment.h"

#include "ast/identifier.h"
#include "exprnode_inc.h"

// Add a new identifier to the environment.
void Environment::add(nir::Identifier const &Id,
                      std::unique_ptr<nir::ValueNode> Val) {
  Env[Id] = std::move(Val);
}

// Lookup an identifier in the environment.
std::unique_ptr<nir::ValueNode> Environment::lookup(nir::Identifier const &Id) {
  auto It = Env.find(Id);
  if (It != Env.end()) {
    return std::make_unique<nir::ValueNode>(*It->second);
  }
  return nullptr;
}
