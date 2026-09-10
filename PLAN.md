# PLAN — issue #122: M2/GC S6, booleans become immediates

## 1. Problem restated

Today every `#t`/`#f` the interpreter touches is a heap-allocated
`ast::BooleanLiteral` (`AST.h:387`), cloned on every literal evaluation
(`Interpreter::visit(ast::BooleanLiteral const&)`, `Interpreter.cpp:752`) and
compared by downcasting (`IfBranch::step`, `Interpreter.cpp:183`). S6 (docs/
value-model-gc-migration.md §4, Phase 2) replaces that allocation with the
frozen `nr_value` immediate encoding (`NR_TRUE`/`NR_FALSE`, `nora_rt.h:40-41`)
for the boolean-literal fast path: a literal `#t`/`#f` becomes a bare
`nr_value` word carried in the machine's `Value` register/`Result`, `IfBranch`
branches on it via `nr_truthy` without allocating, and the public seam
(`Interpreter::getResult()`, consumed by `write()`) materialises a real
`ast::BooleanLiteral` view on demand so external output is unchanged. This is
a CHAR (behaviour-preserving) slice: the observable printed/tested behaviour
of every existing program does not change, only how a boolean literal's value
is represented while it flows through the machine. The forcing assertion
(stated in the issue) is that a linklet whose result is `#f` leaves the raw
`nr_value` immediate `NR_FALSE` in the interpreter's result register — not a
heap object — provable only below the `getResult()` seam.

`Value` (`src/include/Value.h`) is the existing S2/S3 scaffolding handle: it
currently holds *either* an exclusively-owned legacy `ast::ValueNode`
(`Legacy`) *or* a shared reference into an environment binding (`Shared`).
This slice adds a third, allocation-free alternative — a bare `nr_value`
word — used only for boolean literals in S6 (S7 will extend it to
char/void/null/eof, S8 to fixnums, per the ladder).

## 2. Files to touch

- `src/include/Value.h` — add the immediate-word alternative, its accessors,
  and the materialize-on-demand path used by every existing legacy consumer.
- `src/Interpreter.cpp`:
  - `visit(ast::BooleanLiteral const &Bool)` (~line 752) — deliver the
    immediate instead of cloning.
  - `step(Frame::IfBranch &K)` (~line 183) — branch via `nr_truthy` on the
    immediate fast path; keep the existing `dyn_cast_or_null<BooleanLiteral>`
    as a fallback for values that never became immediates (see §4/§6).
  - `visit(ast::Linklet const &Linklet)` (~lines 105-125) — stop routing the
    per-form result through `unique_ptr<ast::ValueNode>` (`Last`); carry it as
    a `Value` end-to-end so an immediate produced by the last top-level form
    survives into `Result` unmaterialized.
  - `applyProcedure` (~line 415) — `if (!Op) { abortEval(); ... }` must keep
    treating an immediate-boolean operator as "engaged" (so `(#t 2)` still
    falls through to the "expected a procedure" diagnostic instead of being
    silently misreported as an aborted/missing value). This falls out of the
    `Value::operator bool()` fix in Value.h; no logic change needed here, but
    it is a regression risk to verify explicitly (§5, new test).
- `src/include/Interpreter.h` — add a small test-only accessor (next to
  `getPeakKont()`/`getGCHeapSize()`, ~line 88) exposing the raw immediate
  behind `Result`, e.g. `std::optional<nr_value> getResultImmediate() const`.
  This is the only way to observe the forcing invariant, since `getResult()`
  deliberately hides it.
- `test/unit/test_interpreter.cpp` — extend the `Run` helper (§ the S5
  seam, `runLinklet`) to also capture `I.getResultImmediate()` before
  `getResult()` is even called is not required once `getResult()` is made
  non-mutating (see §3 slice 1); capture it alongside `result` regardless, so
  the forcing assertion is one `Run` field read, matching S5's "one seam, one
  place" pattern. Add new `TEST_CASE`s (§3).
- `test/integration/ifcond2.rkt` (new) — `(if #f 1 2)`, the first integration
  coverage of a *false* literal condition (`ifcond.rkt`/`ifcond1.rkt` only
  cover a true literal and a truthy bound value).
- `test/integration/error-noncallable-boolean.rkt` (new) — `(#f 1 2)` still
  reports `application: expected a procedure in operator position` (today's
  only such test, `error-noncallable.rkt`, applies an `Integer`, not a
  boolean, so it does not exercise the `operator bool()` path this slice
  touches).
- No `src/mlir/`, `src/include/nir/`, `expander/expander.rktl`, or `docs/`
  changes.

## 3. TDD slices

1. **RED → GREEN: the forcing test.**
   Test: new `TEST_CASE("#f literal result is the NR_FALSE immediate, not an
   allocated BooleanLiteral", "[interp][m2][gc]")` in `test/unit/
   test_interpreter.cpp`. Runs `(linklet () () #f)` through `Run`/
   `runLinklet`, asserts `R.rawImmediate == NR_FALSE` (a new `Run` field —
   see §2) *and* keeps the existing-shape `R.expectBool(false)` so both the
   internal representation and the external seam are pinned in one place.
   RED because `getResultImmediate()`/the `Value` immediate alternative don't
   exist yet — the test won't compile/link.
   Production code: add the `nr_value`-word alternative to `Value` (§4 for
   the exact shape and the move-safety fix), `visit(BooleanLiteral)` delivers
   it, `visit(Linklet)` threads `Value` (not `unique_ptr<ValueNode>`) through
   `Last`, `Interpreter::getResultImmediate()` reads `Result` without
   materializing it, and `getResult()` special-cases `Result.isImmediate()`
   by constructing an `ast::BooleanLiteral` view directly (never calling
   `Result.get()`, so `Result` is left untouched and `getResultImmediate()`
   gives the same answer whether called before or after `getResult()`).

2. **RED → GREEN: IfBranch's fast path.**
   Test: extend `TEST_CASE` for `ifcond.rkt`/add `test/integration/
   ifcond2.rkt` — `(if #f 1 2)` → `CHECK: 2` (closes the coverage gap: no
   existing test drives `IfBranch` with a literal `#f` condition). Also a
   unit test pinning the *fallback* path so it isn't accidentally deleted:
   `(let-values ([(x) #f]) (if x 1 2))` → `2` — here `x` is bound via
   `Environment`/`toShared()`, which still materializes a legacy
   `BooleanLiteral` (§4), so this specifically exercises
   `dyn_cast_or_null<BooleanLiteral>`, not `nr_truthy`.
   Production code: rewrite `step(Frame::IfBranch &K)` to check
   `Val.isImmediate()` first (`nr_truthy` on the raw word, no allocation);
   only when that's false does it fall back to today's
   `takeLegacy()` + `dyn_cast_or_null<ast::BooleanLiteral>` path.

3. **RED → GREEN: legacy call boundaries still work for an immediate
   argument.**
   Tests: unit `TEST_CASE`s for `(eq? #t #t)`, `(eq? #t #f)` (currently
   *no* test passes a bare boolean literal to any `RuntimeFunction` — `eq?`
   is only exercised on boxes/pairs/symbols, per the existing suite) —
   these would segfault today if `Value::get()` returned `nullptr` for an
   engaged immediate, because `EqFunction`/`BoxFunction`/etc. dereference
   `Args[i]` unconditionally. Also add `(box #t)` / `(unbox (box #t))` →
   `#t` as a second, independent boundary (container construction, not just
   predicate dispatch).
   Production code: `Value::get()`, `Value::takeLegacy()`, and
   `Value::toShared()` each materialize a fresh `ast::BooleanLiteral` into
   `Legacy` on demand when only the immediate alternative is engaged (guarded
   by `assert(W == NR_TRUE || W == NR_FALSE)` — see §6, risk R1). This is the
   "legacy alternative is simply materialized at the boundary" scaffolding
   rule from docs/value-model-gc-migration.md §3, applied to every existing
   consumer that expects a non-null `ast::ValueNode*` for an engaged `Value`,
   not just `getResult()`.

4. **RED → GREEN: `operator bool()` / non-callable diagnostic.**
   Test: new `test/integration/error-noncallable-boolean.rkt` — `(#f 1 2)` →
   `CHECK: error: application: expected a procedure in operator position`.
   Without this fix, an immediate-only `Value` is falsy under today's
   `operator bool() { return Legacy || Shared; }`, so `applyProcedure`'s
   `if (!Op) { abortEval(); return; }` would silently abort instead of
   reporting the diagnostic — a real regression, not a hypothetical one, and
   currently untested (`error-noncallable.rkt` only applies an `Integer`).
   Production code: `Value::operator bool()` also checks the immediate
   alternative.

5. **Final slice: sanitizer/asan/ubsan confirmation (no new test, a gate).**
   Run `ctest --preset asan` and `ctest --preset ubsan` in addition to
   `debug`. This slice touches `Value`'s ownership/lifetime logic (a new
   scalar alternative plus lazy materialization into `Legacy`), which is
   exactly the kind of change §6/R8 of the migration doc calls out for
   sanitizer verification, even though nothing here is GC-allocated yet.

Each slice above is its own commit (small, individually reviewable, and the
repo has squash merges disabled so they stay as the durable history).

## 4. Design of the `Value` immediate alternative

Add a third alternative to `Value` (`src/include/Value.h`) alongside
`Legacy`/`Shared`:

```cpp
nr_value Imm = 0; // 0 is not a valid nr_value under any tag — "unengaged"
```

(`#include "nora_rt.h"` needed in `Value.h`.) Rationale for a raw scalar with
a `0` sentinel over `std::optional<nr_value>`: `Value`'s move constructor/
assignment are currently `= default`, which is correct for `unique_ptr`/
`shared_ptr` members (their own move ops reset the source) but is *not*
correct for `std::optional` (its move constructor copies the engaged state
and does **not** reset the source) — that would leave a moved-from `Value`
still reporting `isImmediate() == true`, silently breaking the "moving out
empties the handle" contract the header comment already documents for
`Legacy`. A raw `nr_value Imm = 0` needs the same care (the implicit move of
a scalar member is a copy, not a reset), so `Value`'s move ctor/assignment
must become hand-written either way:

```cpp
Value(Value &&Other) noexcept
    : Legacy(std::move(Other.Legacy)), Shared(std::move(Other.Shared)),
      Imm(std::exchange(Other.Imm, 0)) {}
Value &operator=(Value &&Other) noexcept {
  Legacy = std::move(Other.Legacy);
  Shared = std::move(Other.Shared);
  Imm = std::exchange(Other.Imm, 0);
  return *this;
}
```

API additions:
- `static Value immediate(nr_value W)` — factory, mirrors `Value::share(...)`.
- `bool isImmediate() const { return Imm != 0; }`
- `nr_value rawImmediate() const { assert(isImmediate()); return Imm; }`
- `explicit operator bool() const` gains `Imm != 0 ||` (§3 slice 4).
- `get()`, `takeLegacy()`, `toShared()` each materialize into `Legacy` first
  when `Imm != 0 && !Legacy && !Shared` (§3 slice 3), via one private member
  helper (not a free function — this repo's clang-tidy setup rejects new
  file-scope free functions in a diff-scoped anon-namespace check; keep it a
  `Value` member):
  ```cpp
  void materializeLegacy() {
    if (Imm != 0 && !Legacy && !Shared) {
      assert(Imm == NR_TRUE || Imm == NR_FALSE); // only booleans in S6
      Legacy = std::make_unique<ast::BooleanLiteral>(nr_truthy(Imm));
      Imm = 0;
    }
  }
  ```
  Making `get()` non-const (auditing call sites first — see below) avoids
  needing `mutable` members; if some call site genuinely requires `get()`
  to stay `const`, fall back to `mutable Legacy`/`mutable Imm` instead of
  changing the signature everywhere.
- `getResult()` does **not** go through this path: it special-cases
  `Result.isImmediate()` up front and constructs the `ast::BooleanLiteral`
  view directly (reading `Result.rawImmediate()`, never calling
  `Result.get()`), so `Result` is left unmutated and `getResultImmediate()`
  gives a stable answer regardless of call order relative to `getResult()`.

Before writing `get()` as non-const, grep confirms every existing call site
(`Vals[I].get()`, `Op.get()` in `applyProcedure`, `bindValues`' by-value
`Value Val` parameter) already operates on a non-const `Value`, so the
signature change should be call-site-transparent — verify this holds at
implementation time in case new call sites were added since this plan was
written.

## 5. AST/visitor surface

No new `ASTNodeKind` and no visitor signature changes. `ast::BooleanLiteral`
(`AST_BooleanLiteral`, `AST.h:48`) stays exactly where it is in the enum —
it remains the AST node for the `#t`/`#f` *literal syntax* (parsed by
`Parse::parseBooleanLiteral`, still constructed for `QuotedExpr` payloads,
still what `write()` prints). Every `ASTVisitor`/`Interpreter` `visit(...)`
overload list is untouched; only the *body* of
`Interpreter::visit(ast::BooleanLiteral const&)` changes (what it delivers),
not its signature or its place in the alphabetically-sorted visitor list.
`AnalysisFreeVars::visit(ast::BooleanLiteral const&)` (`AnalysisFreeVars.cpp:
140`) is pure AST traversal (already a no-op: "boolean literals have no free
variables") and is unaffected by how the *runtime value* is represented.

## 6. Risk areas

- **R1 — `materializeLegacy()` must never fire for a non-boolean immediate.**
  S6 only ever constructs `Value::immediate(W)` with `W ∈ {NR_TRUE,
  NR_FALSE}`; the `assert(Imm == NR_TRUE || Imm == NR_FALSE)` in
  `materializeLegacy()` is the guard that stops S7 (char/void/null/eof) or
  S8 (fixnums) from silently materializing the wrong AST node type if a
  future slice starts constructing other immediates before updating this
  function — it must grow a real dispatch (or its own per-kind helper) at
  that point, not silently fall through.
- **R2 — the fallback path in `IfBranch` is load-bearing, not dead code.**
  `zero?`/`eq?`/`continuation-mark-set-first` (`Runtime.cpp:144,201,318`)
  still return heap `ast::BooleanLiteral`s through the *untouched*
  `RuntimeFunction::operator()` ABI (`const ValueNode*` in, `unique_ptr
  <ValueNode>` out — that migration is out of scope here, see §7), and any
  `#t`/`#f` that round-trips through `Environment`/`toShared()` (e.g.
  `(define x #f)` then `(if x ...)`) also materializes. So the existing
  tail-loop test driving `(if (zero? n) ...)` and the new
  `let-values`-bound-boolean test (§3 slice 2) both exercise
  `dyn_cast_or_null<ast::BooleanLiteral>`, not `nr_truthy` — both paths must
  stay correct and are both covered by tests in this plan.
- **R3 — `operator bool()` conflates two different "boolean-ish" concepts.**
  `Value`'s `operator bool()` means "is this handle engaged" (a C++-level
  null check — an aborted evaluation's `Value(nullptr)` is the only
  currently-existing falsy case), *not* Racket truthiness (a `Value` holding
  `BooleanLiteral(false)` is C++-engaged/true but Racket-false). The
  immediate alternative must be folded into the *engaged* check
  (`Imm != 0`, which is true for both `NR_TRUE` and `NR_FALSE`) — never into
  a Racket-truthiness check — or `applyProcedure`'s non-callable diagnostic
  breaks (§3 slice 4) while an unrelated bug (treating `#f` as "no value")
  would be introduced elsewhere if this distinction were blurred.
- **R4 — `write()`'s Racket-compatible output.** Zero risk by construction:
  `write()` is never called on anything but a real, materialized
  `ast::BooleanLiteral` (either the pre-existing legacy path, or the new
  `getResult()` special case), so its `CHECK:`-asserted output
  (`test/integration/ifcond.rkt` et al.) cannot change.
  `QuotedExpr`'s payload (e.g. `'#t`) never reaches
  `Interpreter::visit(BooleanLiteral)` at all — `visit(QuotedExpr)`
  (`Interpreter.cpp:792`) clones the whole quoted node directly — so
  `test/integration/quote11.rkt`/`quote12.rkt` are unaffected.
- **R5 — memory safety.** No GC allocation is introduced in this slice
  (Phase 2 of the ladder is explicitly "no allocation, no roots" — the
  immediate word needs no rooting under either of §2's two mechanisms). The
  only lifetime change is `Value`'s hand-written move ops (§4) replacing
  `= default`; run `debug`, `asan`, and `ubsan` presets (§3 slice 5) to catch
  any use-after-move or double-materialize mistake, per docs/
  value-model-gc-migration.md §6 R8's general caution about half-migrated
  ownership changes being invisible to casual review.

## 7. Out of scope

- **Extending immediate-preservation through `Call`/`WcmMark` frames.**
  `step(Frame::Call&)` and `step(Frame::WcmMark&)` currently do
  `unique_ptr<ValueNode> V = Val.takeLegacy(); Kont.pop_back();
  deliver(std::move(V));` — a pointless round-trip that (after §3 slice 3's
  `takeLegacy()` change) would materialize an immediate the moment it exits
  a lambda call or a `with-continuation-mark` result, e.g.
  `((lambda () #f))` would *not* stay `NR_FALSE` through `Call::step`. This
  is a real, mechanical follow-up (swap the round-trip for
  `Value V = std::move(Val); Kont.pop_back(); deliver(std::move(V));`) but
  is deliberately **not** bundled here: it's a distinct, independently
  reviewable simplification, not required by the issue's stated forcing
  test (a bare top-level `#f` never touches `Call`/`WcmMark`), and bundling
  it risks conflating "make literals immediate" with "make call/mark
  plumbing allocation-free," which the issue text does not ask for.
- **Converting `RuntimeFunction`-produced booleans (`eq?`, `zero?`,
  `continuation-mark-set-first`) to immediates at the `applyProcedure`
  `deliver(std::move(R))` boundary.** Those functions operate over the
  legacy `const ValueNode*`/`unique_ptr<ValueNode>` ABI (`Runtime.h`), which
  this slice does not touch; teaching `applyProcedure` to detect "R is a
  `BooleanLiteral`" and re-wrap it as an immediate would work but expands
  this CHAR slice into an ABI-boundary change the issue doesn't describe.
  Left as a follow-up alongside the `RuntimeFunction` ABI migration
  (untracked by name in the current ladder).
- **S7 (char/void/null/eof → immediates) and S8 (fixnums → `nr_fixnum`).**
  Explicitly the next slices in the ladder; this plan does not touch `Char`,
  `Void`, `Integer`, or their AST/runtime representations.
- **Any `docs/`, MLIR (`src/mlir/`, `src/include/nir/`), or
  `expander/expander.rktl`** changes — none are implicated.
- **Formatting/lint-only cleanup** of files this slice happens to touch,
  beyond what `clang-format`/`clang-tidy` require for the new lines.
