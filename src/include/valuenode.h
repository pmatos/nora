#pragma once

#include <variant>

#include "exprnode.h"

namespace nir {

class Integer;
class Values;
class Void;

using ValueNode = std::variant<Integer, Values, Void>;

struct ToExprNode {
  std::unique_ptr<ExprNode> operator()(nir::Integer &&Int);
  std::unique_ptr<ExprNode> operator()(nir::Values &&V);
  std::unique_ptr<ExprNode> operator()(nir::Void &&Vd);
};

}; // namespace nir