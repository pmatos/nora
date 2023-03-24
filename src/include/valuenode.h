#pragma once

#include <variant>

#include "exprnode.h"

namespace nir {

class Integer;
class Values;
class Void;
class List;
class Lambda;

using ValueNode = std::variant<Integer, Values, Void, List, Lambda>;

struct ToExprNode {
  std::unique_ptr<ExprNode> operator()(nir::Integer &&Int);
  std::unique_ptr<ExprNode> operator()(nir::Values &&V);
  std::unique_ptr<ExprNode> operator()(nir::Void &&Vd);
  std::unique_ptr<ExprNode> operator()(nir::List &&Lst);
  std::unique_ptr<ExprNode> operator()(nir::Lambda &&L);
};

}; // namespace nir