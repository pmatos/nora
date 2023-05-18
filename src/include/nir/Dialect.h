#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "nora.h"

#include "nir/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// nir operations.
#define GET_OP_CLASSES
#include "nir/NirOps.h.inc"
