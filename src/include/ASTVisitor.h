#pragma once

#include <memory>

#include "ast_fwd.h"

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;
  virtual void visit(ast::Identifier const &Id) = 0;
  virtual void visit(ast::Integer const &Int) = 0;
  virtual void visit(ast::Linklet const &Linklet) = 0;
  virtual void visit(ast::DefineValues const &DV) = 0;
  virtual void visit(ast::Values const &V) = 0;
  virtual void visit(ast::Void const &Vd) = 0;
  virtual void visit(ast::Lambda const &L) = 0;
  virtual void visit(ast::Closure const &L) = 0;
  virtual void visit(ast::Begin const &B) = 0;
  virtual void visit(ast::List const &L) = 0;
  virtual void visit(ast::Application const &A) = 0;
  virtual void visit(ast::SetBang const &SB) = 0;
  virtual void visit(ast::IfCond const &If) = 0;
  virtual void visit(ast::BooleanLiteral const &Bool) = 0;
  virtual void visit(ast::LetValues const &LV) = 0;
  virtual void visit(ast::RuntimeFunction const &RF) = 0;
};