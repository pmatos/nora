#pragma once

#include <memory>
#include <vector>

#include "../exprnode.h"
#include "identifier.h"

namespace nir {

class DefineValues {
public:
  DefineValues(std::vector<Identifier> Ids, std::unique_ptr<ExprNode> &Body);
  DefineValues(const DefineValues &DV) = delete;
  DefineValues(DefineValues &&DV) = default;
  DefineValues &operator=(const DefineValues &DV) = delete;
  DefineValues &operator=(DefineValues &&DV) = default;
  ~DefineValues() = default;

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

  private:
    std::vector<Identifier>::const_iterator BeginIt, EndIt;
  };

  IdRange getIds() const { return IdRange{Ids}; }
  [[nodiscard]] const ExprNode &getBody() const;
  [[nodiscard]] size_t countIds() const { return Ids.size(); }

private:
  std::vector<Identifier> Ids;
  std::unique_ptr<ExprNode> Body;
};

}; // namespace nir