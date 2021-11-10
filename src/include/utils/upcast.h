#pragma once

#include <memory>

#include "exprnode.h"
#include "toplevelnode.h"
#include "valuenode.h"

std::unique_ptr<nir::TLNode> upcastNode(std::unique_ptr<nir::ExprNode> &E);
std::unique_ptr<nir::ExprNode> upcastNode(std::unique_ptr<nir::ValueNode> &D);