#pragma once

#include <memory>

#include "exprnode.h"

namespace nir {

class IfCond {
public:
  IfCond() = default;
  IfCond(const IfCond &);
  IfCond(IfCond &&) = default;
  IfCond &operator=(const IfCond &) = delete;
  IfCond &operator=(IfCond &&I) = default;
  ~IfCond() = default;

  void setCond(std::unique_ptr<ExprNode> &&Cond);
  void setThen(std::unique_ptr<ExprNode> &&Then);
  void setElse(std::unique_ptr<ExprNode> &&Else);

  [[nodiscard]] ExprNode const &getCond() const;
  [[nodiscard]] ExprNode const &getThen() const;
  [[nodiscard]] ExprNode const &getElse() const;

private:
  std::unique_ptr<ExprNode> Cond;
  std::unique_ptr<ExprNode> Then;
  std::unique_ptr<ExprNode> Else;
};
}; // namespace nir