#include "Valueify.h"

#include <llvm/Support/ErrorHandling.h>

#include "AST.h"
#include "ASTRuntime.h"
#include "Casting.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Valueifier"

void Valueify::visit(ast::Application const &A) {
  auto V = std::make_unique<ast::List>();

  for (auto &Expr : A.getExprs()) {
    Expr->accept(*this);
    V->appendExpr(std::move(Result));
  }
  Result = dyn_castU<ast::ValueNode, ast::List>(V);
}

void Valueify::visit(ast::Begin const &B) {
  auto V = std::make_unique<ast::List>();

  for (auto &Expr : B.getBody()) {
    Expr->accept(*this);
    V->appendExpr(std::move(Result));
  }
  Result = dyn_castU<ast::ValueNode, ast::List>(V);
}

void Valueify::visit(ast::BooleanLiteral const &Bool) {
  Result = std::unique_ptr<ast::ValueNode>(Bool.clone());
}

void Valueify::visit(ast::Closure const &C) {
  Result = std::unique_ptr<ast::ValueNode>(C.clone());
}

void Valueify::visit(ast::DefineValues const &DV) {
  auto V = std::make_unique<ast::List>();

  auto Ids = std::make_unique<ast::List>();
  for (auto &Id : DV.getIds()) {
    Id.accept(*this);
    Ids->appendExpr(std::move(Result));
  }
  V->appendExpr(std::move(Ids));

  DV.getBody().accept(*this);
  V->appendExpr(std::move(Result));
  Result = dyn_castU<ast::ValueNode, ast::List>(V);
}

void Valueify::visit(ast::Identifier const &Id) {
  LLVM_DEBUG(llvm::dbgs() << "Valueify::visit(ast::Identifier const &Id)\n");
  Result = std::unique_ptr<ast::ValueNode>(new ast::Symbol(Id.getName()));
}
void Valueify::visit(ast::IfCond const &If) {}
void Valueify::visit(ast::Integer const &Int) {
  Result = std::unique_ptr<ast::ValueNode>(Int.clone());
}
void Valueify::visit(ast::Lambda const &L) {}
void Valueify::visit(ast::LetValues const &LV) {}
void Valueify::visit(ast::Linklet const &Linklet) {}
void Valueify::visit(ast::List const &L) {
  Result = std::unique_ptr<ast::ValueNode>(L.clone());
}
void Valueify::visit(ast::QuotedExpr const &L) {}
void Valueify::visit(ast::RuntimeFunction const &LV) {}
void Valueify::visit(ast::SetBang const &SB) {}
void Valueify::visit(ast::Values const &V) {
  Result = std::unique_ptr<ast::ValueNode>(V.clone());
}
void Valueify::visit(ast::Void const &Vd) {
  Result = std::unique_ptr<ast::ValueNode>(Vd.clone());
}
void Valueify::visit(ast::Symbol const &Sym) {
  Result = std::unique_ptr<ast::ValueNode>(Sym.clone());
}