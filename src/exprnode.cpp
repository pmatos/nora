#include "exprnode.h"

#include "ast/arithplus.h"
#include "ast/definevalues.h"
#include "ast/identifier.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/values.h"
#include "ast/void.h"

std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::Identifier &&Id) {
  return std::make_unique<nir::TLNode>(Id);
}
std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::Integer &&Int) {
  return std::make_unique<nir::TLNode>(Int);
}
std::unique_ptr<nir::TLNode> nir::ToTopLevelNode::operator()(nir::Values &&V) {
  return std::make_unique<nir::TLNode>(std::move(V));
}
std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::ArithPlus &&AP) {
  return std::make_unique<nir::TLNode>(std::move(AP));
}
std::unique_ptr<nir::TLNode> nir::ToTopLevelNode::operator()(nir::Void &&Vd) {
  return std::make_unique<nir::TLNode>(std::move(Vd));
}
std::unique_ptr<nir::TLNode> nir::ToTopLevelNode::operator()(nir::Lambda &&L) {
  return std::make_unique<nir::TLNode>(std::move(L));
}