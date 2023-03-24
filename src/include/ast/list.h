#pragma once

#include <cstdint>
#include <vector>

#include "valuenode.h"

// Defines the AST of Linklet programs
// Access to the nodes is done using the visitor pattern.
namespace nir {

class List {
public:
  List() = default;
  List(List const &L);
  List(List &&L) = default;
  List &operator=(const List &L) = delete;
  List &operator=(List &&L) = default;
  ~List() = default;

  void appendExpr(std::unique_ptr<ValueNode> &&Expr);
  [[nodiscard]] size_t length() const { return Values.size(); }
  [[nodiscard]] ValueNode const &operator[](size_t I) const;

private:
  std::vector<std::unique_ptr<nir::ValueNode>> Values;
};

}; // namespace nir