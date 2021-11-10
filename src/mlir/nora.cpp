#include "nora.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"

#include "nir/Dialect.h.inc"

extern "C" {
void nora_init() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::nir::NIRDialect>();
  mlir::registerAllDialects(registry);
}
} // extern "C"
