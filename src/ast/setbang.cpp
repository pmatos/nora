#include "ast/setbang.h"

#include "exprnode_inc.h"

using namespace nir;

SetBang::SetBang(const SetBang &SB) {
  Id = std::make_unique<Identifier>(*SB.Id);
  Expr = std::make_unique<ExprNode>(*SB.Expr);
}

Identifier const &SetBang::getIdentifier() const { return *Id; }
ExprNode const &SetBang::getExpr() const { return *Expr; }

void SetBang::setIdentifier(std::unique_ptr<Identifier> &&I) {
  Id = std::move(I);
}
void SetBang::setExpr(std::unique_ptr<ExprNode> &&E) { Expr = std::move(E); }
