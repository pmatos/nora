#include "ast/begin.h"

#include "exprnode_inc.h"

using namespace nir;

// Copy Constructor for Begin.
Begin::Begin(const Begin &B) {
  for (const auto &Expr : B.Body) {
    std::unique_ptr<ExprNode> Ptr = std::make_unique<ExprNode>(*Expr);
    Body.emplace_back(std::move(Ptr));
  }
}

void Begin::appendExpr(std::unique_ptr<ExprNode> &&E) {
  Body.emplace_back(std::move(E));
}