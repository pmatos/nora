#pragma once

// This interpreter is based on std::variant to implement the visitor pattern.
// Follows a solution proposed in C++ Software Design by Klaus Iglberger.

#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "AST.h"
#include "ASTVisitor.h"
#include "Environment.h"
#include "Runtime.h"

// The interpreter class uses a visitor pattern to access the nodes.

class Interpreter : public ASTVisitor {
public:
  Interpreter();

  // Note: keep the list sorted alphabetically.
  virtual void visit(ast::Application const &A) override;
  virtual void visit(ast::Begin const &B) override;
  virtual void visit(ast::BooleanLiteral const &Bool) override;
  virtual void visit(ast::Char const &C) override;
  virtual void visit(ast::Closure const &L) override;
  virtual void visit(ast::DefineValues const &DV) override;
  virtual void visit(ast::Identifier const &Id) override;
  virtual void visit(ast::IfCond const &If) override;
  virtual void visit(ast::Integer const &Int) override;
  virtual void visit(ast::Keyword const &K) override;
  virtual void visit(ast::Lambda const &L) override;
  virtual void visit(ast::LetValues const &LV) override;
  virtual void visit(ast::Linklet const &Linklet) override;
  virtual void visit(ast::List const &L) override;
  virtual void visit(ast::QuotedExpr const &L) override;
  virtual void visit(ast::RuntimeFunction const &LV) override;
  virtual void visit(ast::SetBang const &SB) override;
  virtual void visit(ast::String const &Str) override;
  virtual void visit(ast::Symbol const &Sym) override;
  virtual void visit(ast::Values const &V) override;
  virtual void visit(ast::Vector const &Vec) override;
  virtual void visit(ast::Void const &Vd) override;

  // Checks if an identifier is bound in the current environment.
  bool isBound(const ast::Identifier &Id) const;

  // Get the current saved result.
  std::unique_ptr<ast::ValueNode> getResult() const {
    assert(Result && "No result has been recorded during interpretation.");
    return std::unique_ptr<ast::ValueNode>(Result->clone());
  };
  std::unique_ptr<ast::ValueNode>
  callFunction(const std::string &Name,
               const llvm::SmallVector<const ast::ValueNode *> &Args) {
    return Runtime::getInstance().callFunction(Name, Args);
  }

private:
  std::vector<Environment> Envs;          /// Environment map for identifiers.
  std::unique_ptr<ast::ValueNode> Result; /// Result of the last evaluation.
};
