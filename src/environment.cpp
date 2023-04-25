#include "environment.h"

// Add a new identifier to the environment.
void Environment::add(ast::Identifier const &Id,
                      std::unique_ptr<ast::ValueNode> Val) {
  Env[Id] = std::move(Val);
}

// Lookup an identifier in the environment.
std::unique_ptr<ast::ValueNode>
Environment::lookup(ast::Identifier const &Id) const {
  auto It = Env.find(Id);
  if (It != Env.end()) {
    return std::unique_ptr<ast::ValueNode>(It->second->clone());
  }
  return nullptr;
}
