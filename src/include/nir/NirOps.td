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
def NIR_Dialect : Dialect {
  let summary = "NORA IR Dialect";
  let description = [{
    NIR (NORA IR) is specified here as an MLIR dialect.
  }];

  // The namespace of our dialect.
  let name = "nir";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::mlir::nir";
}

// def NIR_NoraType : DialectType<NIR_Dialect, CPred<"$_self.isa<NoraType>()">, "NIR Nora type">;

// def NIR_NoraValueAttr : Attr<CPred<"$_self.isa<NoraValueAttr>()">,
//                                    "NIR Nora Value attribute"> {
//     let storageType = [{ NoraValueAttr }];
//     let returnType = [{ nr_value_t * }];
// }

// Base class for nir dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class NIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<NIR_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// NIR Operations
//===----------------------------------------------------------------------===//

def UnimplementedOp : NIR_Op<"unimplemented"> {
  let summary = "unimplemented operation";
  let description = [{
    unimplemented
  }];

  let results = (outs);
  // let builders = [
  //   OpBuilder<(ins "nr_datatype_t *":$type)>
  // ];
}

// def ConstantOp : NIR_Op<"constant", [ConstantLike, NoSideEffect]> {
//   let summary = "constant";
//   let description = [{
//     Constant operation turns a literal into an SSA value. The data is attached
//     to the operation as an attribute. For example:

//     ```mlir
//       %0 = nir.constant dense<2>
//     ```
//   }];

//   // The constant operation takes an attribute as the only input.
//   let arguments = (ins);

//   // The constant operation returns a single value of NumberType
//   let results = (outs);

//   // Add custom build methods for the constant operation. This method
//   // populates the `state` that MLIR uses to create operations i.e. these
//   // are used when using `builder.create<ConstantOp>(...)`.
//   // let builders = [
//   //   // Build a constant with a given constant numeric value.
//   //   OpBuilder<(ins "nr_value_t *":$value, "nr_datatype_t *":$type)>
//   // ];

//   // Set the folder bit so that we can implement constant folders.
//   //let hasFolder = 1;
// }

// def CallWithValues : NIR_Op<"call-with-values", [ConstantLike, NoSideEffect]> {
// }

// def Lambda : NIR_Op<"lambda", [ConstantLike, NoSideEffect]> {
// }

// def PrintValues : NIR_Op<"print-values", [ConstantLike, NoSideEffect]> {
// }

#endif // NIR_OPS
