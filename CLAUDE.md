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

## Architecture

- **AST hierarchy** (`src/include/AST.h`) — `ASTNode` → `TLNode` → `ExprNode`
  → `ValueNode`. Nodes use LLVM-style RTTI: each carries an `ASTNodeKind` tag
  and a `classof`, checked via `isa`/`cast`/`dyn_cast` (`Casting.h`).
  **The `ASTNodeKind` enum order encodes the hierarchy** — `classof` for each
  level is a range check bounded by the `First_TLNode`/`First_ExprNode`/
  `First_ValueNode` markers — so new node kinds must land in the correct
  region and the enum must stay sorted.
- **`ClonableNode<Derived, Base>`** (CRTP, `AST.h`) implements `clone()`/
  `accept()` for concrete leaf node types; new node classes should derive
  through it rather than hand-rolling those.
- **Visitors** — `ASTVisitor.h` and `Interpreter.h` (and any future visitor)
  each declare one `visit(NodeType const&)` per node kind, and both keep that
  list **alphabetically sorted**. Adding a node kind means adding its `visit`
  overload to every visitor and keeping the ordering.

### `dump()` vs `write()` — not interchangeable

- `dump()` is a **debug** dump to `llvm::dbgs()` (stderr), called
  unconditionally wherever it's invoked (e.g. `main` calls `AST->dump()`
  under `-emit=ast`) — the `-debug` flag only sets `llvm::DebugFlag`, which
  gates the `LLVM_DEBUG()` macro, and nothing in `src/` uses that macro today.
- `write()` is the **user-facing** result printer (`std::cout`/`llvm::outs()`);
  `main` prints the final value via `Result->write()`, and its output must
  match Racket's printed representation — this is what integration-test
  `CHECK:` lines assert against.

## Conventions

- Formatting: `clang-format` (LLVM base style; `.clang-format`). CI fails on any
  drift, so run it before committing.
- Linting: `clang-tidy` (`.clang-tidy`).
- Optional local hooks: `pre-commit install` (see `.pre-commit-config.yaml`).
- Claude Code auto-formats edited C/C++ files via a `PostToolUse` hook in
  `.claude/settings.json` (runs `clang-format-22`, matching the pinned
  pre-commit version; silently no-ops if that binary is absent).

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

## Known limitations

- **`QuotedExpr::operator==` is not implemented.** It unconditionally
  returns `false` (see the `// FIXME` in `AST.cpp`) instead of structurally
  comparing the quoted AST nodes. `write()` itself correctly handles quoted
  dotted lists and vectors (`test/integration/quote11.rkt`, `quote12.rkt`
  both pass).
