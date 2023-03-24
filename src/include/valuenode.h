#pragma once

#include <variant>

#include "exprnode.h"

namespace nir {

class Integer;
class Values;
class Void;
class List;

using ValueNode = std::variant<Integer, Values, Void, List>;

struct ToExprNode {
  std::unique_ptr<ExprNode> operator()(nir::Integer &&Int);
  std::unique_ptr<ExprNode> operator()(nir::Values &&V);
  std::unique_ptr<ExprNode> operator()(nir::Void &&Vd);
  std::unique_ptr<ExprNode> operator()(nir::List &&Vd);
};

}; // namespace nir