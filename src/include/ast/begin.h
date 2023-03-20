#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"

namespace nir {

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

  size_t bodyCount() const { return Body.size(); }

  void appendExpr(std::unique_ptr<ExprNode> &&E);

private:
  std::vector<std::unique_ptr<ExprNode>> Body;
};

}; // namespace nir