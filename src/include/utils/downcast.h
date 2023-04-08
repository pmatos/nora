#pragma once

#include <memory>

#include "exprnode.h"
#include "valuenode.h"

std::unique_ptr<nir::ValueNode>
downcastExprToValueNode(std::unique_ptr<nir::ExprNode> &&E);