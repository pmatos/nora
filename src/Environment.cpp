#include "Environment.h"

#include "AST.h"
#include "Value.h"

#include <cassert>
#include <utility>

// Add a new identifier to the environment.
void Environment::add(ast::Identifier const &Id, Value Val) {
  // A binding is never engineered to be empty; catch that loudly here rather
  // than have a later lookup silently read back as "unbound".
  assert(Val && "Environment::add given an empty Value");
  Env[Id] = Val.toShared();
}

// Lookup an identifier in the environment. Shares the bound value instead of
// cloning it.
Value Environment::lookup(ast::Identifier const &Id) const {
  auto It = Env.find(Id);
  if (It != Env.end()) {
    return Value::share(It->second);
  }
  return nullptr;
}

EnvPtr envExtend(const EnvPtr &Parent, const Environment &Vars) {
  return std::make_shared<Scope>(Scope{Environment(Vars), Parent});
}

Value envLookup(const EnvPtr &Env, ast::Identifier const &Id) {
  for (const Scope *S = Env.get(); S != nullptr; S = S->Parent.get()) {
    if (auto V = S->Vars.lookup(Id)) {
      return V;
    }
  }
  return nullptr;
}

bool envSet(const EnvPtr &Env, ast::Identifier const &Id, Value Val) {
  for (Scope *S = Env.get(); S != nullptr; S = S->Parent.get()) {
    if (S->Vars.lookup(Id)) {
      S->Vars.add(Id, std::move(Val));
      return true;
    }
  }
  return false;
}
