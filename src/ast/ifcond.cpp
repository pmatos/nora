#include "ast/ifcond.h"

#include "exprnode_inc.h"

using namespace nir;

// Copy Constructor for IfCond.
IfCond::IfCond(const IfCond &I) {
  Cond = std::make_unique<ExprNode>(*I.Cond);
  Then = std::make_unique<ExprNode>(*I.Then);
  Else = std::make_unique<ExprNode>(*I.Else);
}

void IfCond::setCond(std::unique_ptr<ExprNode> &&C) { Cond = std::move(C); }
void IfCond::setThen(std::unique_ptr<ExprNode> &&T) { Then = std::move(T); }
void IfCond::setElse(std::unique_ptr<ExprNode> &&E) { Else = std::move(E); }

ExprNode const &IfCond::getCond() const { return *Cond; }
ExprNode const &IfCond::getThen() const { return *Then; }
ExprNode const &IfCond::getElse() const { return *Else; }
