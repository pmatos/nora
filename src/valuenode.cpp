#include "valuenode.h"

#include <memory>

#include "ast/booleanliteral.h"
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
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::List &&Lst) {
  return std::make_unique<nir::ExprNode>(std::move(Lst));
}
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Lambda &&L) {
  return std::make_unique<nir::ExprNode>(std::move(L));
}
std::unique_ptr<nir::ExprNode>
nir::ToExprNode::operator()(nir::BooleanLiteral &&Bool) {
  return std::make_unique<nir::ExprNode>(std::move(Bool));
}