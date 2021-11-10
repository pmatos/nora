#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"

namespace nir {

class ArithPlus {
public:
  ArithPlus() = default;
  ArithPlus(const ArithPlus &AP);
  ArithPlus(ArithPlus &&AP) = default;
  ArithPlus &operator=(const ArithPlus &AP) = delete;
  ArithPlus &operator=(ArithPlus &&AP) = default;
  ~ArithPlus() = default;

  void appendArg(std::unique_ptr<ExprNode> &&Arg);
  [[nodiscard]] const std::vector<std::unique_ptr<ExprNode>> &getArgs() const;

private:
  std::vector<std::unique_ptr<ExprNode>> Args;
};
}; // namespace nir