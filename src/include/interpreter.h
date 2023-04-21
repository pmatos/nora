#pragma once

// This interpreter is based on std::variant to implement the visitor pattern.
// Follows a solution proposed in C++ Software Design by Klaus Iglberger.

#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "ASTVisitor.h"
#include "ast.h"
#include "environment.h"

// The interpreter class uses a visitor pattern to access the nodes.

class Interpreter : public ASTVisitor {
public:
  Interpreter();

  virtual void visit(ast::Identifier const &Id) override;
  virtual void visit(ast::Integer const &Int) override;
  virtual void visit(ast::Linklet const &Linklet) override;
  virtual void visit(ast::DefineValues const &DV) override;
  virtual void visit(ast::Values const &V) override;
  virtual void visit(ast::ArithPlus const &AP) override;
  virtual void visit(ast::Void const &Vd) override;
  virtual void visit(ast::Lambda const &L) override;
  virtual void visit(ast::Begin const &B) override;
  virtual void visit(ast::List const &L) override;
  virtual void visit(ast::Application const &A) override;
  virtual void visit(ast::SetBang const &SB) override;
  virtual void visit(ast::IfCond const &If) override;
  virtual void visit(ast::BooleanLiteral const &Bool) override;
  virtual void visit(ast::LetValues const &LV) override;

  // Get the current saved result.
  std::unique_ptr<ast::ValueNode> getResult() const {
    return std::unique_ptr<ast::ValueNode>(Result->clone());
  };

private:
  std::vector<Environment> Envs;          /// Environment map for identifiers.
  std::unique_ptr<ast::ValueNode> Result; /// Result of the last evaluation.
};
