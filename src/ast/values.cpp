#include "ast/values.h"

#include "exprnode_inc.h"

using namespace nir;

Values::Values(std::vector<std::unique_ptr<ExprNode>> Exprs)
    : Exprs(std::move(Exprs)) {}

// Copy constructor for values.
Values::Values(const Values &V) {
  for (const auto &Expr : V.getExprs()) {
    std::unique_ptr<ExprNode> Ptr = std::make_unique<ExprNode>(*Expr);
    Exprs.emplace_back(std::move(Ptr));
  }
}

Values::ExprRange::ExprRange(
    std::vector<std::unique_ptr<nir::ExprNode>>::const_iterator EsBegin,
    std::vector<std::unique_ptr<nir::ExprNode>>::const_iterator EsEnd)
    : BeginIt(EsBegin), EndIt(EsEnd) {}

ExprNode const &Values::ExprRange::operator[](size_t I) const {
  return *BeginIt[I];
}