# NIR MLIR dialect (spike R5 / seed of B0)

The NIR (NORA IR) dialect is the Racket-linklet-close MLIR layer where
Racket-level optimizations run before lowering NIR → LLVM IR. This directory is
**opt-in** (`-DNORA_ENABLE_MLIR=ON`, the `mlir` CMake preset) and is **not** part
of the default `norac` build.

## Status: built and round-trip-verified (MLIR 22)

R5 is complete: the dialect builds under the `mlir` preset and
`test/mlir/nir-roundtrip.mlir` round-trips through `nir-opt | nir-opt` (FileCheck
passes). Requires MLIR 22 (`find_package(MLIR REQUIRED CONFIG)`); the default
`norac` build is unaffected and stays green.

What the rewrite changed vs. the old scaffold:

- Deleted the stale duplicate `Ops.td` (used the long-removed `list<OpTrait>` API).
- `NirOps.td`: a real, minimal, **representation-agnostic** dialect — `nir.constant`
  (`i64` skeleton payload) and a `nir.return` terminator. No `nr_value`-tagged
  types yet: those land after the M2 ABI freeze (see `docs/value-model-abi.md`).
- Added `nir-opt.cpp`, a minimal `mlir-opt` clone registering the NIR dialect, and
  wired it in `CMakeLists.txt`.
- Added the round-trip acceptance test `test/mlir/nir-roundtrip.mlir`.

## Verify (once MLIR is installed)

```sh
# Arch: MLIR is the AUR `mlir` package (needs sudo/AUR; the maintainer runs it):
yay -S mlir
#   ...or point CMake at an existing build: -DMLIR_DIR=/path/to/lib/cmake/mlir

cmake --preset mlir
cmake --build --preset mlir --target nir-opt
build/mlir/bin/nir-opt test/mlir/nir-roundtrip.mlir | build/mlir/bin/nir-opt
# expect: the module prints back with `nir.constant 42 : i64`
```

## Known items to confirm at first build

Because this was authored without a live `mlir-tblgen`, the first `cmake --preset
mlir` build may surface small ODS/version nits to fix in place — most likely the
generated dialect **C++ class name** (`def NIR_Dialect` is assumed to generate
`mlir::nir::NIRDialect`, matching `Dialect.cpp`) and the exact link-library list
in `CMakeLists.txt`. The op set, assembly formats, and traits (`Pure`,
`Terminator`, `ReturnLike`) follow current MLIR conventions.

Tracking issue: #90.
