#include "nir/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::nir;

#include "nir/Dialect.cpp.inc"

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void NirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "nir/Ops.cpp.inc"
      >();
}

//
// IntConstantOp
//

// Built a constant operation.
// The builder is passed as an argumnet, so is the tstate tha tthis method
// is expected to fill in order to build the opreation.
void IntConstantOp::build(mlir::OpBuilder &Builder, mlir::OperationState &State,
                          double value) {
  IntConstantOp::build();
}

//
// TableGen'd op method definitions
//

#define GET_OP_CLASSES
#include "nir/Ops.cpp.inc"