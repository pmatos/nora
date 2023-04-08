#include "ast/letvalues.h"

#include <utility>

#include "exprnode_inc.h"

using namespace nir;

LetValues::LetValues(const LetValues &DV) {
  for (auto const &Id : DV.Ids) {
    Ids.emplace_back(Id);
  }
  for (auto const &Expr : DV.Exprs) {
    Exprs.emplace_back(std::make_unique<nir::ExprNode>(*Expr));
  }
  for (auto const &Expr : DV.Body) {
    Body.emplace_back(std::make_unique<nir::ExprNode>(*Expr));
  }
}

void LetValues::appendBinding(std::vector<Identifier> &&Ids,
                              std::unique_ptr<ExprNode> Expr) {
  this->Ids.emplace_back(std::move(Ids));
  this->Exprs.emplace_back(std::move(Expr));
}

void LetValues::appendBody(std::unique_ptr<ExprNode> Expr) {
  Body.emplace_back(std::move(Expr));
}
nir::LetValues::IdRange LetValues::getBindingIds(size_t Idx) const {
  assert(Idx < Ids.size());
  return IdRange{Ids[Idx]};
}
ExprNode const &LetValues::getBindingExpr(size_t Idx) const {
  return *Exprs[Idx];
}
ExprNode const &LetValues::getBodyExpr(size_t Idx) const { return *Body[Idx]; }
size_t nir::LetValues::exprsCount() const { return Exprs.size(); }
