#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"

namespace nir {

class Application {
public:
  Application() = default;
  Application(const Application &);
  Application(Application &&) = default;
  Application &operator=(const Application &) = delete;
  Application &operator=(Application &&I) = default;
  ~Application() = default;

  void appendExpr(std::unique_ptr<ExprNode> &&Expr);
  [[nodiscard]] size_t length() const { return Exprs.size(); }
  [[nodiscard]] ExprNode const &operator[](size_t I) const;

private:
  std::vector<std::unique_ptr<ExprNode>> Exprs;
};
}; // namespace nir