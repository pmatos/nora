//===- nir-opt.cpp - NIR optimizer / round-trip driver --------------------===//
//
// A minimal `mlir-opt` clone that registers the NIR dialect alongside the
// upstream dialects, so .mlir files using `nir.*` can be parsed, verified, and
// printed. This is the R5 spike's verification tool (parse->print round-trip)
// and the seed of B0's opt driver.
//
//===----------------------------------------------------------------------===//
#include "nir/Dialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::nir::NIRDialect>();
  mlir::registerAllDialects(registry); // func, builtin, arith, ...
  mlir::registerAllPasses();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "NIR optimizer / round-trip driver\n", registry));
}
