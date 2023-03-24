#include "ast/list.h"

#include "exprnode_inc.h"
#include "utils/upcast.h"
#include "valuenode.h"

nir::List::List(List const &L) {
  for (auto &V : L.Values) {
    Values.emplace_back(std::make_unique<ValueNode>(*V));
  }
}

void nir::List::appendExpr(std::unique_ptr<ValueNode> &&Value) {
  Values.emplace_back(std::move(Value));
}

nir::ValueNode const &nir::List::operator[](size_t I) const {
  return *Values[I];
}