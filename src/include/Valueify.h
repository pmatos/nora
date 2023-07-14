#pragma once

#include "AST.h"
#include "ASTVisitor.h"

#include <llvm/ADT/STLExtras.h>

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <vector>

// File implementing valueification.
// This transforms an AST into a value AST.

class Valueify : public ASTVisitor {
public:
  // Note: keep the list sorted alphabetically.
  virtual void visit(ast::Application const &A) override;
  virtual void visit(ast::Begin const &B) override;
  virtual void visit(ast::BooleanLiteral const &Bool) override;
  virtual void visit(ast::Closure const &C) override;
  virtual void visit(ast::DefineValues const &DV) override;
  virtual void visit(ast::Identifier const &Id) override;
  virtual void visit(ast::IfCond const &If) override;
  virtual void visit(ast::Integer const &Int) override;
  virtual void visit(ast::Lambda const &L) override;
  virtual void visit(ast::LetValues const &LV) override;
  virtual void visit(ast::Linklet const &Linklet) override;
  virtual void visit(ast::List const &L) override;
  virtual void visit(ast::QuotedExpr const &L) override;
  virtual void visit(ast::RuntimeFunction const &LV) override;
  virtual void visit(ast::SetBang const &SB) override;
  virtual void visit(ast::Symbol const &Sym) override;
  virtual void visit(ast::Values const &V) override;
  virtual void visit(ast::Void const &Vd) override;

  // Get the current saved result.
  std::unique_ptr<ast::ValueNode> getResult() const {
    assert(Result && "Result is null!");
    return std::unique_ptr<ast::ValueNode>(Result->clone());
  };

private:
  std::unique_ptr<ast::ValueNode> Result;
};
