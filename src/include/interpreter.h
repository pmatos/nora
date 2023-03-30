#pragma once

// This interpreter is based on std::variant to implement the visitor pattern.
// Follows a solution proposed in C++ Software Design by Klaus Iglberger.

#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "ast/identifier.h"
#include "astnode.h"
#include "environment.h"
#include "valuenode.h"

// The interpreter class takes over ownership of the AST nodes.

class Interpreter {
public:
  std::unique_ptr<nir::ValueNode> operator()(nir::Identifier const &Id);
  std::unique_ptr<nir::ValueNode> operator()(nir::Integer const &Int);
  std::unique_ptr<nir::ValueNode> operator()(nir::Linklet const &Linklet);
  std::unique_ptr<nir::ValueNode> operator()(nir::DefineValues const &DV);
  std::unique_ptr<nir::ValueNode> operator()(nir::Values const &V);
  std::unique_ptr<nir::ValueNode> operator()(nir::ArithPlus const &AP);
  std::unique_ptr<nir::ValueNode> operator()(nir::Void const &Vd);
  std::unique_ptr<nir::ValueNode> operator()(nir::Lambda const &L);
  std::unique_ptr<nir::ValueNode> operator()(nir::Begin const &B);
  std::unique_ptr<nir::ValueNode> operator()(nir::List const &L);
  std::unique_ptr<nir::ValueNode> operator()(nir::Application const &A);
  std::unique_ptr<nir::ValueNode> operator()(nir::SetBang const &SB);
  std::unique_ptr<nir::ValueNode> operator()(nir::IfCond const &If);
  std::unique_ptr<nir::ValueNode> operator()(nir::BooleanLiteral const &Bool);

private:
  // Environment map for identifiers.
  std::vector<Environment> Envs;
};
