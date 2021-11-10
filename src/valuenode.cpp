#include "valuenode.h"

#include <memory>

#include "ast/arithplus.h"
#include "ast/identifier.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/values.h"
#include "ast/void.h"

std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Integer &&Int) {
  return std::make_unique<nir::ExprNode>(Int);
}
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Values &&V) {
  return std::make_unique<nir::ExprNode>(std::move(V));
}
std::unique_ptr<nir::ExprNode> nir::ToExprNode::operator()(nir::Void &&Vd) {
  return std::make_unique<nir::ExprNode>(std::move(Vd));
}