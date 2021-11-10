#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"

namespace nir {

class Values {
public:
  explicit Values(std::vector<std::unique_ptr<ExprNode>> Exprs);
  Values(const Values &V);
  Values(Values &&V) = default;
  Values &operator=(const Values &V) = delete;
  Values &operator=(Values &&V) = default;

  ~Values() = default;

  class ExprRange {
  public:
    ExprRange() = delete;
    ExprRange(std::vector<std::unique_ptr<ExprNode>>::const_iterator EsBegin,
              std::vector<std::unique_ptr<ExprNode>>::const_iterator EsEnd);
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    [[nodiscard]] ExprNode const &operator[](size_t I) const;

  private:
    std::vector<std::unique_ptr<ExprNode>>::const_iterator BeginIt, EndIt;
  };

  [[nodiscard]] ExprRange getExprs() const {
    return {Exprs.cbegin(), Exprs.cend()};
  }
  [[nodiscard]] size_t countExprs() const { return Exprs.size(); }

private:
  std::vector<std::unique_ptr<ExprNode>> Exprs;
};
}; // namespace nir