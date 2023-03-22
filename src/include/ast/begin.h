#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"

namespace nir {

// AST Node representing a begin or begin0 expression.
class Begin {
public:
  Begin() = default;
  Begin(const Begin &B);
  Begin(Begin &&B) = default;
  Begin &operator=(const Begin &B) = delete;
  Begin &operator=(Begin &&B) = default;
  ~Begin() = default;

  [[nodiscard]] const std::vector<std::unique_ptr<ExprNode>> &getBody() const {
    return Body;
  }
  [[nodiscard]] size_t bodyCount() const { return Body.size(); }
  [[nodiscard]] bool isZero() const { return Zero; }

  void appendExpr(std::unique_ptr<ExprNode> &&E);
  void markAsBegin0() { Zero = true; }

private:
  std::vector<std::unique_ptr<ExprNode>> Body;
  bool Zero = false;
};

}; // namespace nir