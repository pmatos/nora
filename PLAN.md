# Plan — issue #121: M2/GC S5 — test-seam helpers

## 1. Problem restated

`test/unit/test_interpreter.cpp` currently asserts on interpreter results by
reaching directly for the runtime-value representation at ~20 call sites:
`llvm::dyn_cast<ast::Integer>(...)` followed by `REQUIRE(*Int == N)`, and
`llvm::dyn_cast<ast::BooleanLiteral>(...)` followed by
`REQUIRE(B->value() == ...)`. Per `docs/value-model-gc-migration.md` §4 (S18),
the migration's final slice flips this public seam from materializing an
`ast::Integer`/`BooleanLiteral` view to reading `nr_value` words
(`nr_fixnum_val`/`nr_truthy`) directly. Today that flip would require editing
~20 scattered call sites. S5 is a pure test-refactor (`CHAR`, no production
code, no behavior change) that introduces `expectInt`/`expectBool`/
`expectResult` wrappers so the eventual flip is a one-place edit in S18. No
new test behavior is added; every existing `REQUIRE` stays semantically
equivalent.

## 2. Files to touch

- `test/unit/test_interpreter.cpp` — the only file touched. No production
  code (`src/`, `src/include/`), no `CMakeLists.txt` changes (no new files,
  no new dependencies), no `test/integration/` changes.

## 3. Design constraint discovered during planning (read first)

`.clang-tidy` enables both `misc-use-anonymous-namespace` (wants file-scope
free functions moved into an anonymous namespace or marked `static`) and
`llvm-prefer-static-over-anonymous-namespace` (wants anonymous-namespace free
functions marked `static` instead). These directly contradict each other for
**any new file-scope free function**, and `.github/workflows/clang-tidy.yml`
is diff-scoped with `-warnings-as-errors`, so a brand-new free function (in
either form) fails CI on its own added line — this bit a prior session on
this exact repo. The existing helpers (`runLinklet`, `tailLoopPeak`, etc.) are
grandfathered only because they sit on unchanged lines.

**Resolution:** do not add new namespace-scope free functions. Add
`expectInt`/`expectBool`/`expectResult` as members (static where no `Run`
instance is available, non-static where one is) of the existing `Run` struct
at `test/unit/test_interpreter.cpp:20-23`, which already lives inside the
file's anonymous namespace. Class member functions are outside the scope of
both checks regardless of the enclosing namespace, so this sidesteps the
conflict entirely instead of picking a side that still fails.

**Verified constraint on that resolution:** `.clang-tidy`'s
`llvm-prefer-static-over-anonymous-namespace` has an
`AllowMemberFunctionsInClass` option (default `true`) that exempts member
functions **defined inline in the class body** but still flags an
out-of-line `Run::expectInt(...) { ... }` definition sitting in the
anonymous namespace. This repo's `.clang-tidy` does not override that
option (confirmed: `grep AllowMemberFunctionsInClass .clang-tidy` — no
match, so the default applies), so the design is only safe if **every**
helper is defined inline, inside the struct body. No out-of-line member
definitions.

Concretely, extend `Run` (all bodies inline, per the constraint above):

```cpp
struct Run {
  bool ok = false;
  std::unique_ptr<ast::ValueNode> result;

  // The seam S18 will rewrite: downcast to a materialized ValueNode view.
  // Localized here so the eventual nr_value read replaces one definition.
  template <typename T> static const T *expectResult(const ast::ValueNode *Node) {
    REQUIRE(Node);
    const auto *Downcast = llvm::dyn_cast<T>(Node);
    REQUIRE(Downcast);
    return Downcast;
  }

  static void expectInt(const ast::ValueNode *Node, int64_t Expected) {
    REQUIRE(*expectResult<ast::Integer>(Node) == Expected);
  }

  static void expectBool(const ast::ValueNode *Node, bool Expected) {
    REQUIRE(expectResult<ast::BooleanLiteral>(Node)->value() == Expected);
  }

  void expectInt(int64_t Expected) const {
    REQUIRE(ok);
    expectInt(result.get(), Expected);
  }

  void expectBool(bool Expected) const {
    REQUIRE(ok);
    expectBool(result.get(), Expected);
  }
};
```

Call sites keyed off a `Run` use the instance form (`R.expectInt(42)`); the
two call sites that downcast a raw `ValueNode*`/`unique_ptr<ValueNode>` not
wrapped in a `Run` (the direct `Runtime::callFunction` results) use the
static form (`Run::expectBool(Result.get(), false)`).

`expectResult<ast::List>` is used once directly (not through a fourth
`expectList` wrapper — the issue names exactly three helpers, and `List` is
the same generic primitive instantiated at a third type): the call site
keeps its existing explicit `REQUIRE(R.ok)` (the static `expectResult`
overload does not check `ok`, only that the node is non-null and downcasts —
see §6), then calls `Run::expectResult<ast::List>(R.result.get())`, and the
list-element check becomes `Run::expectInt(&(*L)[0], 0)`.

## 4. TDD slices

This is a `CHAR` (characterization) refactor per the migration doc: there is
no new behavior, so there is no RED step. Each slice is a small, independently
buildable, independently reviewable commit; the whole suite (both Catch2
mains + the `lit`/FileCheck integration corpus) must stay green under
`debug`, `asan`, and `ubsan` after every slice, per the migration plan's
per-slice promise.

1. **Add `expectResult`/`expectInt`, convert the `Integer`-result call
   sites.**
   Extend `Run` with the `expectResult` template and both `expectInt`
   overloads from §3 (leave `expectBool` for slice 2, so nothing lands
   unused), and convert every call site that currently does
   `dyn_cast<ast::Integer>(...)` + `REQUIRE(*Int == N)` on a `Run` result to
   `R.expectInt(N)`. Current sites (line numbers as of this plan; they will
   shift as edits land — relocate by pattern, not line number, once slice 1
   is committed): 47-49, 120-122, 134-136, 161-163, 168-170, 180-182,
   279-281, 353-355. Add `#include <cstdint>` (new use of `int64_t` on a
   changed line; `misc-include-cleaner` is enabled — see §6). Build
   `test_interpreter` and run its suite (see the build/test commands below),
   confirm identical pass/fail behavior to before the edit.

2. **Add `expectBool`, convert the `BooleanLiteral`-result call sites.**
   Extend `Run` with both `expectBool` overloads from §3, and convert every
   `dyn_cast<ast::BooleanLiteral>(...)` + `REQUIRE(B->value())` /
   `REQUIRE_FALSE(B->value())` on a `Run` result to `R.expectBool(true)` /
   `R.expectBool(false)`. Current sites: 145-147, 152-154, 190-192, 197-199,
   208-210, 216-218, 259-261. Rebuild, rerun the suite.

3. **Convert the two raw-`ValueNode*` call sites and the `List` element
   check.** No new helpers needed — both already exist from slices 1-2.
   - The `eq?`-unwraps-a-quoted-symbol test (lines 237-242, 247-252) calls
     `Runtime::getInstance().callFunction(...)` directly (no `Run` wrapper);
     convert its two `dyn_cast<ast::BooleanLiteral>(Result.get())` +
     `REQUIRE_FALSE(...->value())` checks to
     `Run::expectBool(Result.get(), false)`.
   - The WCM-replaces-not-accumulates test (lines 322-334): keep the
     existing explicit `REQUIRE(R.ok)`, replace
     `auto *L = llvm::dyn_cast<ast::List>(R.result.get()); REQUIRE(L);` with
     `auto *L = Run::expectResult<ast::List>(R.result.get());` (drop the now
     redundant `REQUIRE(L)` — `expectResult` already asserts the downcast
     succeeded), keep `REQUIRE(L->length() == 1)` as-is, and convert the
     element check `dyn_cast<ast::Integer>(&(*L)[0])` + `REQUIRE(*Elem ==
     0)` to `Run::expectInt(&(*L)[0], 0)`.
   Rebuild, rerun the full suite to close out the slice set.

**Build/test commands** (cap parallelism per the operating contract; no
production code changes in `src/`, so only the `test_interpreter` target is
on the critical path — it links `AST.cpp`/`ASTRuntime.cpp`/`Interpreter.cpp`
etc., so it still needs a build, just not a full `norac` one):

```
cmake --build --preset debug --target test_interpreter -j<N>
./build/debug/bin/test_interpreter          # per slice (1, 2, 3)
```

Run the sanitizer presets once, after slice 3, not per slice — this is a
test-only refactor with no ownership/lifetime changes in `src/`, so the
per-slice signal that matters is "the rewritten assertions still pass";
`asan`/`ubsan` confirm the refactor didn't introduce a mismatched-type
downcast or otherwise misbehave under stricter runtime checks, which is a
property of the finished refactor, not of each intermediate step:

```
cmake --build --preset asan  --target test_interpreter -j<N> && ./build/asan/bin/test_interpreter
cmake --build --preset ubsan --target test_interpreter -j<N> && ./build/ubsan/bin/test_interpreter
```

## 5. AST/visitor surface

Not applicable. No `ASTNode` kind is added, changed, or removed; no visitor
(`ASTVisitor`, `Interpreter`) gains or loses a `visit()` overload. This slice
is test-only.

## 6. Risk areas

- **clang-tidy dual-check conflict (see §3).** The concrete risk is
  reintroducing a file-scope free function by accident (e.g. drafting
  `expectInt` as a helper next to `runLinklet` instead of as a `Run` member).
  Mitigation: the member-function design in §3, plus running
  `clang-tidy-diff.py` locally against `/usr/bin/clang-tidy` (v22, not the
  `~/.local/bin` v11 shadow) on the diff before committing, matching how the
  prior session verified this exact constraint.
- **`misc-include-cleaner`.** New lines introduce `int64_t` (needs
  `<cstdint>`) and continue using `llvm::dyn_cast`/`ast::Integer`/
  `ast::BooleanLiteral`/`ast::List` (already included via `AST.h` and
  `llvm/Support/Casting.h`, so no change needed there). Verify no new
  transitively-included-only symbol is used.
- **clang-format drift.** Run `/usr/bin/clang-format -i` (v22, matching CI
  and the pinned pre-commit hook) on the changed file before each commit;
  the `.claude/settings.json` `PostToolUse` hook should also catch this
  automatically on each edit.
- **Behavior-preservation / Catch2 message decomposition.** `REQUIRE(*Int ==
  Expected)` and `REQUIRE(Value == Expected)` are binary-expression asserts,
  whereas today's `REQUIRE(S->value())` / `REQUIRE_FALSE(D->value())` are
  unary. Pass/fail outcomes are identical; only the printed operands in a
  *failing* assertion's diagnostic differ (both now show the actual and
  expected value, which is a strict improvement, not a behavior change to
  guard against). No functional risk.
- **No `write()`/printed-output risk, no free-variable-analysis risk, no
  interpreter memory-safety risk.** This slice does not touch `src/`; the
  `debug`/`asan`/`ubsan` reruns in §4 are to confirm the test-file refactor
  itself introduced no regression (e.g. a mismatched downcast type), not
  because production code's ownership/lifetime is in play.
- **Line-number drift across slices.** Each slice in §4 changes the file, so
  line numbers cited for later slices are approximate at plan-writing time;
  the implementer should re-locate call sites by pattern (`dyn_cast<ast::
  Integer>`, `dyn_cast<ast::BooleanLiteral>`) rather than trusting exact line
  numbers after slice 1 lands.

## 7. Out of scope

- **The actual S18 seam flip** (rewriting `expectInt`/`expectBool` to read
  `nr_fixnum_val`/`nr_truthy` off an `nr_value` instead of downcasting an
  `ast::ValueNode`). That is blocked on S6-S17 landing first (per the slice
  ladder's `Depends-on` chain) and is explicitly future work this issue only
  prepares for.
- **Any production code change** in `src/`/`src/include/` — S5 is
  test-file-only per the issue and the migration doc.
- **Adding `expectResult` instantiations for value kinds not currently
  exercised** by `test_interpreter.cpp` (e.g. `String`, `Symbol`, `Void`) —
  add them in whichever future slice first needs to assert on that kind, not
  speculatively here.
- **Refactoring `runLinklet`/`tailLoopPeak`/`nonTailLoopPeak`/
  `wcmTailLoopPeak`** or any other existing helper — unrelated to the
  downcast-assertion seam this issue targets.
- **`test/unit/CMakeLists.txt` changes** — no new files or dependencies are
  introduced.
