#include "nir/MLIRGen.h"

#include "ast.h"
#include "nir/Dialect.h"

using namespace mlir::nir;
using namespace nir;

namespace nir {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &Context,
                                          Linklet &LinkletAST) {
  return MLIRGenImpl(Context).mlirGen(LinkletAST);
}

} // namespace nir