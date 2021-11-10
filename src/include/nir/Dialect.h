#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "nora.h"

#include "nir/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// nir operations.
#define GET_OP_CLASSES
#include "nir/NirOps.h.inc"
