# NORA — agent guide

NORA is an experimental Racket implementation in **C++23** with an **LLVM 22**
backend. Today `norac` parses a linklet (a Racket [Fully Expanded
Program](https://docs.racket-lang.org/reference/syntax-model.html#%28part._fully-expanded%29))
and interprets it; the LLVM/MLIR compilation backend is future work. See
`README.md` for the project vision and roadmap.

## Build, run, test

The build is driven by [`CMakePresets.json`](CMakePresets.json). The default
build depends only on **LLVM and GMP** — no MLIR.

```
cmake --preset debug         # configure (also: release, asan, ubsan, coverage, mlir)
cmake --build --preset debug # build -> build/debug/bin/{norac,nora-lit}
ctest --preset debug         # unit (Catch2) + integration (lit/FileCheck) tests
```

- Run the interpreter: `./build/debug/bin/norac test/integration/arithplus.rkt`
- Integration tests only: `./build/debug/bin/nora-lit test/integration -v`
- The code is **warning-clean on both GCC and Clang** and built with
  warnings-as-errors (`-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF` to relax locally).

## Layout

- `src/` — the interpreter. Pipeline: `SourceStream` → `Lex` → `Parse` →
  `AST` (`ast.cpp`/`ast.h`) → `Interpreter` (an `ASTVisitor`) → `Runtime`
  values. Supporting pieces: `Environment`, `ASTRuntime`, `AnalysisFreeVars`,
  `IdPool`, `UTF8`. `main.cpp` wires it together with LLVM `cl::opt`.
- `src/include/Casting.h` — LLVM-style RTTI (`isa`/`cast`/`dyn_cast`) for AST
  nodes; `ASTVisitor.h` — the visitor interface.
- `src/mlir/` + `src/include/nir/` — the **experimental NIR MLIR dialect**.
  This is an empty scaffold, **opt-in** behind `-DNORA_ENABLE_MLIR=ON` (the
  `mlir` preset), and **not used by the interpreter**. Do not add MLIR to the
  default build path.
- `test/unit/` — Catch2 tests (`test_parse.cpp`); Catch2 is fetched by CMake.
- `test/integration/` — `.rkt` files run through `lit` + `FileCheck`.
- `expander/expander.rktl` — a large generated Racket artifact; do not edit.

## Conventions

- Formatting: `clang-format` (LLVM base style; `.clang-format`). CI fails on any
  drift, so run it before committing.
- Linting: `clang-tidy` (`.clang-tidy`).
- Optional local hooks: `pre-commit install` (see `.pre-commit-config.yaml`).

## Adding an integration test

Create `test/integration/<name>.rkt`:

```
;; RUN: norac %s | FileCheck %s
;; CHECK: <expected output>
(linklet () () <expression>)
```

`lit` substitutes `norac` with the built binary; `FileCheck` must be on `PATH`.

## Toolchain notes

- Requires CMake >= 3.24, Ninja, LLVM 22 (matching Clang, or GCC >= 13), and
  GMP with its C++ bindings (`gmpxx`).
- CI (`.github/workflows/`) installs LLVM 22 from apt.llvm.org, builds via the
  presets on Ubuntu 24.04, and pins actions to commit SHAs.
