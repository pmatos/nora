#pragma once

#include "ASTVisitor.h"
#include "ast.h"

#include <llvm/ADT/STLExtras.h>

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <vector>

// File implementing free variable analysis for expressions.

class AnalysisFreeVars : public ASTVisitor {
public:
  virtual void visit(ast::Identifier const &Id) override;
  virtual void visit(ast::Integer const &Int) override;
  virtual void visit(ast::Linklet const &Linklet) override;
  virtual void visit(ast::DefineValues const &DV) override;
  virtual void visit(ast::Values const &V) override;
  virtual void visit(ast::Void const &Vd) override;
  virtual void visit(ast::Lambda const &L) override;
  virtual void visit(ast::Closure const &L) override;
  virtual void visit(ast::Begin const &B) override;
  virtual void visit(ast::List const &L) override;
  virtual void visit(ast::Application const &A) override;
  virtual void visit(ast::SetBang const &SB) override;
  virtual void visit(ast::IfCond const &If) override;
  virtual void visit(ast::BooleanLiteral const &Bool) override;
  virtual void visit(ast::LetValues const &LV) override;
  virtual void visit(ast::RuntimeFunction const &LV) override;

  // Get the current saved result.
  std::set<ast::Identifier> getResult() const { return Result; };

private:
  std::set<ast::Identifier> Result; /// List of free variables.
  llvm::SmallVector<std::set<ast::Identifier>>
      Vars; /// Environment map for identifiers.
};
