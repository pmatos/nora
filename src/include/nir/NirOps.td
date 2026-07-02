//===- NirOps.td - NIR dialect operation definitions -------*- tablegen -*-===//
//
// NIR (NORA IR): an MLIR dialect kept close to Racket linklets so that
// Racket-level optimizations happen here, before lowering NIR -> LLVM IR.
//
// R5 / B0 scope: a *minimal, representation-agnostic* skeleton — just enough to
// round-trip through nir-opt and prove the MLIR toolchain integrates. Concrete
// nr_value-tagged types (!nir.fixnum, !nir.box, !nir.closure, ...) are added
// after the M2 ABI freeze (docs/value-model-abi.md); they are absent here on
// purpose.
//
//===----------------------------------------------------------------------===//

#ifndef NIR_OPS
#define NIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def NIR_Dialect : Dialect {
  let name = "nir";
  let summary = "NORA IR — a Racket-linklet-close MLIR dialect";
  let description = [{
    NIR (NORA IR) models Racket's fully-expanded / linklet layer as an MLIR
    dialect so that Racket-level optimizations (primitive inlining, fixnum
    unboxing, known-call devirtualization, dead-export elimination) can run
    before lowering to LLVM IR.
  }];
  let cppNamespace = "::mlir::nir";
}

// Base class for NIR ops. NOTE: `list<Trait>` is the current ODS spelling; the
// old scaffold's `list<OpTrait>` was removed from MLIR years ago.
class NIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<NIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def NIR_ConstantOp : NIR_Op<"constant", [Pure]> {
  let summary = "materialise a constant";
  let description = [{
    Representation-agnostic skeleton constant. The payload is a builtin integer
    attribute for now; the nr_value-tagged constant lands after the M2 ABI
    freeze (docs/value-model-abi.md).

    ```mlir
    %0 = nir.constant 42 : i64
    ```
  }];
  let arguments = (ins I64Attr:$value);
  let results = (outs I64:$result);
  let assemblyFormat = "$value attr-dict";
}

def NIR_ReturnOp : NIR_Op<"return", [Pure, Terminator, ReturnLike]> {
  let summary = "return terminator";
  let description = [{
    Returns zero or more values from a NIR region. Defined here for B0's
    lowering; it is exercised once a NIR region op (nir.func / nir.linklet)
    exists to host it.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // NIR_OPS
