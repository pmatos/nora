#pragma once

#include <cassert>
#include <memory>
#include <vector>

#include "../exprnode.h"
#include "identifier.h"

namespace nir {

class LetValues {
public:
  LetValues() = default;
  LetValues(const LetValues &DV);
  LetValues(LetValues &&DV) = default;
  LetValues &operator=(const LetValues &DV) = delete;
  LetValues &operator=(LetValues &&DV) = default;
  ~LetValues() = default;

  // View over the Ids
  // FIXME: we should be able to use C++20 view_interface here
  // although my initial attempt failed.
  class IdRange {
  public:
    IdRange() = delete;
    IdRange(const std::vector<Identifier> &Ids)
        : BeginIt(Ids.cbegin()), EndIt(Ids.cend()) {}
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    [[nodiscard]] Identifier const &operator[](size_t Idx) const {
      return *(BeginIt + Idx);
    }

  private:
    std::vector<Identifier>::const_iterator BeginIt, EndIt;
  };

  IdRange getBindingIds(size_t Idx) const;
  ExprNode const &getBindingExpr(size_t Idx) const;
  ExprNode const &getBodyExpr(size_t Idx) const;

  void appendBinding(std::vector<Identifier> &&Ids,
                     std::unique_ptr<ExprNode> Expr);
  void appendBody(std::unique_ptr<ExprNode> Expr);

  size_t bindingCount() const {
    assert(Ids.size() == Exprs.size());
    return Ids.size();
  }
  size_t idsCount() const { return Ids.size(); }
  size_t exprsCount() const;
  size_t bodyCount() const { return Body.size(); }

private:
  std::vector<std::vector<Identifier>> Ids;
  std::vector<std::unique_ptr<ExprNode>> Exprs;
  std::vector<std::unique_ptr<ExprNode>> Body;
};

}; // namespace nir