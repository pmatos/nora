#include "exprnode.h"

#include "ast/booleanliteral.h"
#include "toplevelnode_inc.h"

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
std::unique_ptr<nir::TLNode> nir::ToTopLevelNode::operator()(nir::Begin &&B) {
  return std::make_unique<nir::TLNode>(std::move(B));
}
std::unique_ptr<nir::TLNode> nir::ToTopLevelNode::operator()(nir::List &&L) {
  return std::make_unique<nir::TLNode>(std::move(L));
}
std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::Application &&A) {
  return std::make_unique<nir::TLNode>(std::move(A));
}
std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::SetBang &&SB) {
  return std::make_unique<nir::TLNode>(std::move(SB));
}
std::unique_ptr<nir::TLNode> nir::ToTopLevelNode::operator()(nir::IfCond &&If) {
  return std::make_unique<nir::TLNode>(std::move(If));
}
std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::BooleanLiteral &&Bool) {
  return std::make_unique<nir::TLNode>(std::move(Bool));
}
std::unique_ptr<nir::TLNode>
nir::ToTopLevelNode::operator()(nir::LetValues &&LV) {
  return std::make_unique<nir::TLNode>(std::move(LV));
}