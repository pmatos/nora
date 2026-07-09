set(LLVM_TARGET_DEFINITIONS NirOps.td)
mlir_tablegen(NirOps.h.inc -gen-op-decls)
mlir_tablegen(NirOps.cpp.inc -gen-op-defs)
# -dialect=nir: NirOps.td transitively includes the builtin dialect (via
# BuiltinTypes.td), so the dialect generators must be told which to emit.
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=nir)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=nir)
add_public_tablegen_target(NirIncGen)
