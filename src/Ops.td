//===- Ops.td - NIR dialect operation definitions ----------*- tablegen -*-===//
//
// Defines the operations of the Nora IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef NIR_OPS
#define NIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'nir' dialect in the ODS framework so that we
// can define our operations.
def Nir_Dialect : Dialect {
  let name = "nir";
  let cppNamespace = "::mlir::nir";
}

// Base class for nir dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Nir_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Nir_Dialect, mnemonic, traits>;

#endif // NIR_OPS