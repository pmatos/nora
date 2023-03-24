#include "valuenode.h"

#include <memory>

#include "exprnode_inc.h"

std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Integer &&Int) {
  return std::make_unique<nir::ExprNode>(Int);
}
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Values &&V) {
  return std::make_unique<nir::ExprNode>(std::move(V));
}
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Void &&Vd) {
  return std::make_unique<nir::ExprNode>(std::move(Vd));
}
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::List &&L) {
  return std::make_unique<nir::ExprNode>(std::move(L));
}