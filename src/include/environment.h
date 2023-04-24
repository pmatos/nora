#pragma once

#include <map>
#include <memory>

#include "ast.h"

class Environment {

public:
  Environment() = default;
  Environment(Environment const &) = default;
  Environment(Environment &&) = delete;
  Environment &operator=(Environment const &) = delete;
  Environment &operator=(Environment &&) = delete;

  // Add a new identifier to the environment.
  void add(ast::Identifier const &Id, std::unique_ptr<ast::ValueNode> Val);

  // Lookup an identifier in the environment.
  std::unique_ptr<ast::ValueNode> lookup(ast::Identifier const &Id) const;

private:
  // Environment map for identifiers.
  std::map<ast::Identifier, std::shared_ptr<ast::ValueNode>> Env;
};
