#pragma once

#include <map>
#include <memory>

#include "AST.h"
#include "Value.h"

class Environment {

public:
  Environment() = default;
  Environment(Environment const &) = default;
  Environment(Environment &&) = delete;
  Environment &operator=(Environment const &) = delete;
  Environment &operator=(Environment &&) = delete;

  // Add a new identifier to the environment.
  void add(ast::Identifier const &Id, Value Val);

  // Lookup an identifier in the environment. Shares the bound value rather
  // than cloning it, so repeated lookups of the same identifier alias the
  // same underlying ast::ValueNode.
  Value lookup(ast::Identifier const &Id) const;

  // Drop all bindings. Used to break reference cycles at interpreter teardown
  // (a closure can capture the very scope that binds it).
  void clear() { Env.clear(); }

  // Implement range style access to the Env map.
  auto begin() const { return Env.begin(); }
  auto end() const { return Env.end(); }

private:
  // Environment map for identifiers.
  std::map<ast::Identifier, std::shared_ptr<ast::ValueNode>> Env;
};

// A lexical environment for the abstract machine is a chain of scopes. Each
// Scope holds a flat map of bindings (an Environment) and a pointer to its
// enclosing scope. Lookups walk from the innermost scope outward. Scopes are
// shared via shared_ptr so closures (and, later, captured continuations) can
// share structure with the live machine.
struct Scope;
using EnvPtr = std::shared_ptr<Scope>;
struct Scope {
  Environment Vars;
  EnvPtr Parent;
};

// Create a fresh scope whose bindings are a copy of Vars, enclosed by Parent.
EnvPtr envExtend(const EnvPtr &Parent, const Environment &Vars);

// Look up Id walking the scope chain from Env outward. Returns a Value
// sharing the bound value, or a falsy Value if Id is unbound in the whole
// chain.
Value envLookup(const EnvPtr &Env, ast::Identifier const &Id);

// Mutate the innermost binding of Id in the chain to Val (set! semantics).
// Returns false if Id is unbound anywhere in the chain.
bool envSet(const EnvPtr &Env, ast::Identifier const &Id, Value Val);
