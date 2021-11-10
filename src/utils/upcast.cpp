#include "utils/upcast.h"

#include <variant>

#include "ast/arithplus.h"
#include "ast/definevalues.h"
#include "ast/identifier.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/values.h"
#include "ast/void.h"
#include "exprnode.h"
#include "toplevelnode.h"
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