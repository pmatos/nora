#include "Environment.h"

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

EnvPtr envExtend(const EnvPtr &Parent, const Environment &Vars) {
  return std::make_shared<Scope>(Scope{Environment(Vars), Parent});
}

std::unique_ptr<ast::ValueNode> envLookup(const EnvPtr &Env,
                                          ast::Identifier const &Id) {
  for (const Scope *S = Env.get(); S != nullptr; S = S->Parent.get()) {
    if (auto V = S->Vars.lookup(Id)) {
      return V;
    }
  }
  return nullptr;
}

bool envSet(const EnvPtr &Env, ast::Identifier const &Id,
            std::unique_ptr<ast::ValueNode> Val) {
  for (Scope *S = Env.get(); S != nullptr; S = S->Parent.get()) {
    if (S->Vars.lookup(Id)) {
      S->Vars.add(Id, std::move(Val));
      return true;
    }
  }
  return false;
}
