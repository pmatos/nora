//===- Ops.td - NIR dialect operation definitions ----------*- tablegen -*-===//
//
// Defines the operations of the Nora IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef NIR_OPS
#define NIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'nir' dialect in the ODS framework so that we
// can define our operations.
def NirDialect : Dialect {
  let summary = "NORA IR Dialect";
  let description = [{
    NIR (NORA IR) is specified here as an MLIR dialect.
  }];

  // The namespace of our dialect.
  let name = "nir";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::mlir::nir";
}

// Base class for nir dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class NirOp<string mnemonic, list<Trait> traits = []> :
    Op<NirDialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// NIR Operations
//===----------------------------------------------------------------------===//

// We define a NIR operation by inheriting from our base 'NIR_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def IntConstantOp : NirOp<"int_constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Integer constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = nir.constant 1234567890123456789012345678901234567890 : i1024
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I64:$value);

  // The constant operation returns a single value of IntegerType.
  let results = (outs I64);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

#endif // NIR_OPS
