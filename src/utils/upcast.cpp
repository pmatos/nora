#include "utils/upcast.h"

#include <variant>

#include "exprnode.h"
#include "toplevelnode.h"
#include "toplevelnode_inc.h"
#include "valuenode.h"

// upcastNode moves the unique ptr E of type ExprNode to a unique ptr
// of type TLNode, effectively upcasting the type.
std::unique_ptr<nir::TLNode> upcastNode(std::unique_ptr<nir::ExprNode> &E) {
  if (!E)
    return nullptr;

  nir::ExprNode *EN = E.release();
  nir::ToTopLevelNode toTLNode;
  return std::visit(toTLNode, std::move(*EN));
}

std::unique_ptr<nir::ExprNode> upcastNode(std::unique_ptr<nir::ValueNode> &D) {
  if (!D)
    return nullptr;

  nir::ValueNode *DN = D.release();
  return std::visit(nir::ToExprNode{}, std::move(*DN));
}