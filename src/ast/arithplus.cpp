#include "ast/arithplus.h"

#include "exprnode_inc.h"

using namespace nir;

const std::vector<std::unique_ptr<ExprNode>> &ArithPlus::getArgs() const {
  return Args;
}

void ArithPlus::appendArg(std::unique_ptr<ExprNode> &&Arg) {
  Args.emplace_back(std::move(Arg));
}

// Copy Constructor for ArithPlus.
ArithPlus::ArithPlus(const ArithPlus &AP) {
  for (const auto &Arg : AP.getArgs()) {
    std::unique_ptr<ExprNode> Ptr = std::make_unique<ExprNode>(*Arg);
    Args.emplace_back(std::move(Ptr));
  }
}