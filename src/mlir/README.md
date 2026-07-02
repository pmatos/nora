# NIR MLIR dialect (spike R5 / seed of B0)

The NIR (NORA IR) dialect is the Racket-linklet-close MLIR layer where
Racket-level optimizations run before lowering NIR → LLVM IR. This directory is
**opt-in** (`-DNORA_ENABLE_MLIR=ON`, the `mlir` CMake preset) and is **not** part
of the default `norac` build.

## Status: prepared, NOT yet verified (blocked on MLIR install)

R5's rewrite of the previously broken scaffold is done, but it **has not been
compiled or round-tripped**, because MLIR 22 is not installed on this machine
(LLVM 22 is; MLIR is a separate package). Until MLIR is present, `find_package(MLIR
REQUIRED CONFIG)` fails and none of this builds.

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
