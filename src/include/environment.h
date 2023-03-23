#pragma once

#include <map>
#include <memory>

#include "ast/identifier.h"
#include "valuenode.h"

class Environment {

public:
  Environment() = default;
  Environment(Environment const &) = default;
  Environment(Environment &&) = delete;
  Environment &operator=(Environment const &) = delete;
  Environment &operator=(Environment &&) = delete;

  // Add a new identifier to the environment.
  void add(nir::Identifier const &Id, std::unique_ptr<nir::ValueNode> Val);

  // Lookup an identifier in the environment.
  std::unique_ptr<nir::ValueNode> lookup(nir::Identifier const &Id);

private:
  // Environment map for identifiers.
  std::map<nir::Identifier, std::shared_ptr<nir::ValueNode>> Env;
};
