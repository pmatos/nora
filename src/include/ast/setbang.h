#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"

namespace nir {

class SetBang {
public:
  SetBang() = default;
  SetBang(const SetBang &);
  SetBang(SetBang &&) = default;
  SetBang &operator=(const SetBang &) = delete;
  SetBang &operator=(SetBang &&I) = default;
  ~SetBang() = default;

  void setIdentifier(std::unique_ptr<Identifier> &&Id);
  void setExpr(std::unique_ptr<ExprNode> &&Expr);

  [[nodiscard]] Identifier const &getIdentifier() const;
  [[nodiscard]] ExprNode const &getExpr() const;

private:
  std::unique_ptr<Identifier> Id;
  std::unique_ptr<ExprNode> Expr;
};
}; // namespace nir