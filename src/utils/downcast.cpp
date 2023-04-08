#include "utils/downcast.h"

#include <iostream>
#include <memory>
#include <variant>

#include "exprnode_inc.h"
#include "utils/overloaded.h"
#include "valuenode_inc.h"

std::unique_ptr<nir::ValueNode>
downcastExprToValueNode(std::unique_ptr<nir::ExprNode> &&E) {
  nir::ExprNode *Expr = E.release();
  std::unique_ptr<nir::ValueNode> Val = std::visit(
      overloaded{
          [](nir::Void const &V) {
            return std::make_unique<nir::ValueNode>(V);
          },
          [](nir::Integer const &I) {
            return std::make_unique<nir::ValueNode>(I);
          },
          [](nir::Values const &V) {
            return std::make_unique<nir::ValueNode>(V);
          },
          [](nir::Lambda const &L) {
            return std::make_unique<nir::ValueNode>(L);
          },
          [](nir::BooleanLiteral const &Bool) {
            return std::make_unique<nir::ValueNode>(Bool);
          },
          [](auto const &Err) {
            std::cerr << "Error: Cannot downcast expression that's not a value."
                      << std::endl;
            return std::unique_ptr<nir::ValueNode>();
          }},
      *Expr);
  return Val;
}