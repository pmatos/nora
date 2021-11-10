#include "nir/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::nir;

//===----------------------------------------------------------------------===//
// NirDialect
//===----------------------------------------------------------------------===//
#include "nir/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "nir/NirOps.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void NIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nir/NirOps.cpp.inc"
      >();
}
