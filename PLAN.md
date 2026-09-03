# Plan: issue #119 — M2/GC S3, Value in the environment; share instead of clone-on-lookup

## 1. Problem restated

`Environment::lookup` (and the scope-chain wrapper `envLookup`) currently
returns a *deep clone* of the bound value on every read
(`Environment.cpp:14`, `std::unique_ptr<ast::ValueNode>(It->second->clone())`),
even though the environment's own map already stores a `shared_ptr` per
binding. This is wasteful and, more importantly, is the wrong primitive for
where the value-model migration (`docs/value-model-gc-migration.md`) is
headed: the environment should hand out a *shared reference* to the bound
value, matching what will eventually be a bare GC pointer. This slice (S3 in
the migration doc) routes `Environment::add`/`lookup`/`envLookup`/`envSet`
through the `Value` handle introduced in S2 (PR #118), changes `lookup`'s
implementation from clone to share, and audits every place that currently
assumes a looked-up value is a private, exclusively-owned copy (arithmetic
accumulators, `with-continuation-mark` key/value capture, `set!`) so that
none of them is silently corrupted by aliasing now that two live `Value`
handles can point at the same `ast::ValueNode`.

## 2. Files to touch

- `src/include/Value.h` — add a second, shared-ownership alternative to the
  handle.
- `src/include/Environment.h`, `src/Environment.cpp` — `add`/`lookup` (member
  functions) and `envLookup`/`envSet` (free functions) change signature to
  speak `Value` instead of `std::unique_ptr<ast::ValueNode>`; `lookup`'s body
  changes from `clone()` to a shared copy.
- `src/include/Interpreter.h` — `deliver()`'s parameter type changes from
  `std::unique_ptr<ast::ValueNode>` to `Value`.
- `src/Interpreter.cpp` — no signature changes expected in `visit(Identifier)`
  / `isBound()` beyond what falls out of the above; verify by building.
- `test/unit/test_environment.cpp` (new) — direct unit tests for `Value`'s
  sharing behavior and for `Environment`/`envLookup`/`envSet`.
- `test/unit/CMakeLists.txt` — add `test_environment.cpp` to `test_interpreter`.
- `test/integration/alias-args.rkt` (new), `test/integration/wcm-var-key.rkt`
  (new), `test/integration/arith-self-ref.rkt` (new).
- `test/integration/box.rkt`, `test/integration/pair.rkt` — fix the
  now-inaccurate "the interpreter clones values on every lookup" comment.

No change to `src/include/AST.h`, `ASTNodeKind`, or any `ASTVisitor`/
`Interpreter` `visit()` overload — this slice touches only the runtime value
handle and the environment, not the AST hierarchy.

## 3. Design (read before the slices — this is the part with real risk)

`Environment`'s map already stores `std::map<Identifier,
std::shared_ptr<ast::ValueNode>>` (`Environment.h:33`) — that does **not**
need to change. What needs to change is that `Value` (`Value.h`), currently a
thin move-only wrapper around a single owning `std::unique_ptr<ast::ValueNode>
Legacy`, needs a **second alternative** for values that cross the
`Environment` boundary, so `lookup` can hand one out without cloning:

```cpp
class Value {
  ...
  // NOLINTNEXTLINE(google-explicit-constructor): implicit boundary from legacy.
  template <std::derived_from<ast::ValueNode> T>
  Value(std::unique_ptr<T> V) : Legacy(std::move(V)) {}
  static Value share(std::shared_ptr<ast::ValueNode> V); // new: wraps an
                                                          // environment-shared
                                                          // reference
  std::shared_ptr<ast::ValueNode> toShared();  // new: for Environment::add —
                                                // moves Legacy into a fresh
                                                // shared_ptr, or copies Shared
  std::unique_ptr<ast::ValueNode> takeLegacy(); // existing signature, new body
  ast::ValueNode *get() const;                  // existing signature, new body
  explicit operator bool() const;               // existing signature, new body
private:
  std::unique_ptr<ast::ValueNode> Legacy; // exclusively owned (today's case)
  std::shared_ptr<ast::ValueNode> Shared; // environment-shared (new)
  // Invariant: at most one of {Legacy, Shared} is engaged at a time.
};
```

**Why the existing constructor must become a template.** `Value`'s current
converting constructor is `Value(std::unique_ptr<ast::ValueNode>)` — exactly
one parameter type. C++ allows only **one** user-defined conversion per
implicit sequence, so a call like `deliver(std::make_unique<ast::Void>())`
(`Interpreter.cpp:305`, and similarly `:230,330,345,440,681,769,773`) would
need *two*: `unique_ptr<ast::Void>` → `unique_ptr<ast::ValueNode>` (itself a
converting constructor call), then `unique_ptr<ast::ValueNode>` → `Value`.
That does not compile. The same problem hits `Environment::add`'s callers at
`Interpreter.cpp:530,539` (`std::move(Rest)` is `unique_ptr<ast::List>`,
`Lst` likewise). Making the constructor a template over any
`ast::ValueNode`-derived `T` collapses this to a single user-defined
conversion (`unique_ptr<T>` → `Value` directly), which is what actually makes
the "no call-site changes elsewhere" property in this section true. This
constructor change is additive and belongs in slice 1, with its own RED
test (see §4).

`takeLegacy()`'s new body is the crux:
- If `Legacy` is engaged: `return std::move(Legacy);` — **identical to
  today**, zero extra cost, for every value that never touched an
  environment (arithmetic results, freshly-constructed literals, `deliver`d
  values built with `make_unique`).
- If `Shared` is engaged: the caller needs *exclusive* ownership (it's about
  to store the result in a still-`unique_ptr`-typed `Frame` slot —
  `Frame::Done`/`Saved`/`Callee`/`WcmKeyV` are **not** migrated by this
  slice; that's S4), so materialize a private copy: `auto Owned =
  std::unique_ptr<ast::ValueNode>(Shared->clone()); Shared.reset(); return
  Owned;`.

This is why S3 is a **CHAR** (behavior-preserving) slice, not a performance
win by itself: every current consumption path in `continueStep()` calls
`Val.takeLegacy()` on the very next machine step after a lookup, so the
clone that used to happen inside `Environment::lookup` still happens — it is
just *deferred one call* to the point a not-yet-migrated consumer actually
needs exclusive ownership. The real, immediately-observable wins are the
paths that look up a value and never materialize it:
- `Environment::envSet`'s existence check (`if (S->Vars.lookup(Id))`,
  `Environment.cpp:36`) — today clones a whole value just to test binding
  presence; after this slice it's a `shared_ptr` copy + a boolean check
  (`explicit operator bool()` supports contextual conversion in `if`).
- `Interpreter::isBound` (`Interpreter.cpp:608`) — same pattern.

`deliver()` (`Interpreter.h:177`) changes its parameter type from
`std::unique_ptr<ast::ValueNode>` to `Value`. With the templated constructor
above, **every existing call site of `deliver(...)` that passes a
`unique_ptr<T>` (for any `ast::ValueNode`-derived `T`) keeps compiling
unchanged** — this is what lets `visit(Identifier)` (`Interpreter.cpp:
618-622`) pass the `Value` returned by `envLookup` straight into `deliver`
without an intermediate `takeLegacy()`/clone at the lookup site itself;
materialization happens, if at all, one step later in `continueStep()`.

`Environment::add`/`envSet` take `Value` by value and call `V.toShared()` to
get the `shared_ptr` to store in the map. Every existing call site
(`Vars.add(Ids[0], std::move(Val))`, `Vars.add(LRF.getRestFormal(),
std::move(Rest))` where `Rest` is `unique_ptr<ast::List>`, `envSet(E, *Id,
std::move(V))`, etc.) keeps compiling unchanged via the same templated
conversion — **no call site in `Interpreter.cpp` needs to change for the
`add`/`envSet` *input* direction**, provided the constructor is templated as
above (with the old single-type constructor, `:530` and `:539` would fail to
compile for the same reason `deliver` would).

`envExtend`/`Environment`'s copy constructor are **not** touched: the map's
value type stays `shared_ptr<ast::ValueNode>` (already copyable), so nothing
about `Environment`'s existing copy-constructibility is affected by this
design. (Confirmed `envExtend` has zero call sites today — it is pre-existing
dead code, out of scope to remove here.)

## 4. TDD slices

Slices 1–3 below are each **one commit that must compile and pass on its
own** — `Environment.cpp` is linked into both `test_parse` and
`test_interpreter`, so nothing isolates a half-migrated `Environment` from
the rest of the tree; the three-way split below is what actually keeps every
commit buildable (see the advisor note this plan was checked against).

1. **`Value` gains a shared alternative (additive, `deliver`/`Environment`
   untouched).** RED: new unit tests in `test/unit/test_environment.cpp`
   exercising `Value` in isolation:
   - `Value V = std::make_unique<ast::Integer>(1);` — fails to compile
     against today's single-type constructor; this is the RED that forces
     the template-constructor change in §3.
   - `Value::share(sp)` twice from the same `shared_ptr<ast::Integer>`
     yields two `Value`s whose `get()` return the *same* address.
   - `takeLegacy()` on a `share()`-constructed `Value` returns a
     `unique_ptr` whose `get()` is a *different* address than the original
     `shared_ptr`, but `*result == *original` (via `ast::Integer::
     operator==`).
   - `takeLegacy()` on a `Value` constructed from a plain `unique_ptr`
     (today's path) is unaffected: returns the exact same pointer, no clone.
   - `toShared()` round-trips: a `Value` built from a `unique_ptr`,
     `toShared()`'d, then `Value::share()`'d again, shares identity with a
     second `toShared()` result read from the same slot.
   GREEN: implement the templated constructor, `Shared`, `share()`,
   `toShared()`, and the new `takeLegacy()`/`get()`/`operator bool()` bodies
   in `Value.h`. Nothing outside `Value.h` and its new test file changes in
   this commit — `deliver()` and `Environment` are still exactly as they are
   today, so `test_interpreter`/`test_parse`/`norac` all build and the full
   suite stays green (this commit adds capability nothing yet calls).

2. **`deliver()` speaks `Value`.** RED/GREEN in one step, no new test: change
   `deliver()`'s parameter type to `Value` in `Interpreter.h` (§3). This
   compiles cleanly once slice 1's templated constructor exists (every
   existing `deliver(unique_ptr<T>(...))` call site converts through it
   unchanged), and is a pure no-op behaviorally — `Val = std::move(V)` was
   already doing this same construction implicitly. GREEN: `ctest --preset
   debug` — full suite unchanged, confirming this step really is a no-op
   before slice 3 makes it load-bearing.

3. **`Environment::add`/`lookup`/`envLookup`/`envSet` speak `Value`; `lookup`
   shares.** This is the one commit where `Environment.h` gains `#include
   "Value.h"` and all four signatures change together — they cannot be
   split further without leaving `Environment.cpp` (linked into both test
   binaries) in a non-compiling state. RED: unit tests directly against
   `Environment` and a two-scope chain (construct identifiers via
   `IdPool::instance().create("x")`, values via `std::make_unique<ast::
   Integer>(42)`, scopes via `Scope`/`EnvPtr Parent` directly):
   - `Environment` bound to `x`; two calls to `lookup(x)` return `Value`s
     with the same `get()` address (the actual "kill clone-on-lookup"
     assertion — false against the current `clone()`-based implementation,
     so genuinely RED first).
   - `lookup` of an unbound identifier returns a falsy `Value`.
   - `add` followed by `add` again on the same identifier rebinds (last
     write wins), and a subsequent `lookup` observes the new value.
   - `envLookup` finds a binding in the parent scope from a child `EnvPtr`,
     and repeated `envLookup` calls for the same identifier share identity
     (same assertion, across the scope chain).
   - `envSet` on an identifier bound in the parent mutates the parent's
     binding (observable via a subsequent `envLookup`), and returns `false`
     for an identifier bound nowhere in the chain.
   GREEN: reimplement `Environment::add`/`lookup`/`envLookup`/`envSet` per
   §3 (map type unchanged; only the four function bodies and signatures
   change). Because slice 2 already made `deliver` accept `Value`, and
   slice 1 already made every `unique_ptr<T>`→`Value` call site compile,
   this commit needs no changes to `Interpreter.cpp` beyond what falls out
   of `Environment.h`'s new signatures — verify by building `test_interpreter`,
   `test_parse`, and `norac`, then `ctest --preset debug` for the full suite
   (Catch2 mains + `lit`/FileCheck integration corpus).

4. **Aliasing + audit regression tests.** RED-then-GREEN (these should pass
   immediately once slices 1–4 are correct; if one fails it means the
   materialize-on-demand boundary in `takeLegacy()` has a real bug, not that
   the test is wrong):
   - `test/integration/alias-args.rkt` — the issue's own example: bind a
     `box` once, pass it as both arguments of a two-argument lambda
     (`(f x x)`), mutate through one parameter, read through the other.
     Mirrors the existing `box.rkt`/`pair.rkt` pattern but specifically
     through two *separate* environment lookups of the same identifier
     feeding two different argument slots.
   - `test/integration/wcm-var-key.rkt` — `with-continuation-mark`'s key and
     value expressions are bound identifiers (not literals), confirming the
     mark captures the looked-up value correctly now that `Val.takeLegacy()`
     inside the `WcmKey`/`WcmVal` frame handling may be materializing from a
     `Shared`-alternative `Value` instead of a `Legacy` one.
   - `test/integration/arith-self-ref.rkt` — `(+ x x)`, `(- x x)`,
     `(* x x)` for a single bound integer, pinning that
     `SubtractFunction`'s existing "clone the first arg before `-=`"
     (`Runtime.cpp:52-53`) is unaffected by the two call-site `Args`
     pointers now potentially aliasing the same materialized object.
   - Update the stale "clones on every lookup" comments in
     `test/integration/box.rkt` and `test/integration/pair.rkt` — they
     already describe the *end-state* aliasing correctly, they just
     misattribute *why* it holds (it no longer depends on clone-on-lookup).

5. **Sanitizer gate.** Run `cmake --preset asan && cmake --build --preset
   asan && ctest --preset asan`, then the same for `ubsan`. This is the
   slice most likely to surface a bug in the `Legacy`/`Shared` mutual
   exclusion invariant (e.g. a missed `Shared.reset()` after materializing,
   which would leave both alternatives briefly engaged and could be
   asan-visible if `takeLegacy()` is ever called twice on the same `Value`).
   No new tests; this exercises slices 1–4's tests under the sanitizers.

## 5. AST/visitor surface

Not applicable. This slice adds no `ASTNode`/`ValueNode` kind, so
`ASTNodeKind`'s ordering and every `ASTVisitor`/`Interpreter` `visit()`
overload list are untouched.

## 6. Risk areas

- **Ownership/lifetime (real risk — asan/ubsan required, slice 5).** The new
  `Value::Legacy`/`Shared` mutual-exclusion invariant is the only place this
  slice can introduce a use-after-free or double-materialization. It is
  fully within LSan/ASan's visibility (plain `shared_ptr`/`unique_ptr`, no
  GC involved yet — S3 predates the collector-relevant slices S9+ in the
  migration doc), so the sanitizer presets are a strong, cheap gate.
- **`write()`/printed output — not a risk.** `Interpreter::getResult()`
  (`Interpreter.h:76-81`) already calls `Result.get()->clone()` to produce
  its defensive public-seam copy, and `Result` is populated via
  `Val.takeLegacy()` (`Interpreter.cpp:115,120`), whose *output type*
  (`std::unique_ptr<ast::ValueNode>`) is unchanged by this slice. `write()`
  never observes whether its argument came from `Legacy` or `Shared`.
- **Free-variable analysis — not a risk.** `AnalysisFreeVars` operates on
  the AST's static identifier/binding structure, not on `Environment` or
  `Value`; this slice touches neither.
- **Silent `eq?`/identity drift (R5 in the migration doc) — low risk here.**
  `Box`/`Pair`'s `eq?` already goes through `identity()` (the shared `Cell`
  pointer, `ASTRuntime.h:84,117`), which was already stable across
  clone-on-lookup before this slice (their `clone()` copy-constructs and
  shares the `Cell` — `ASTRuntime.h:70,103`). This slice does not change any
  `eq?`/`equal?` dispatch, so no observable identity behavior changes for
  Racket-level code; the `alias-args.rkt` test pins this explicitly so a
  future slice can't regress it silently.
- **WCM mark capture.** Flagged explicitly by the issue; addressed by the
  `wcm-var-key.rkt` regression test in slice 4 and by the fact that
  `WcmKey`/`WcmVal` frame handling (`Interpreter.cpp:349-398`) already goes
  through `Val.takeLegacy()`, which materializes correctly per §3.
- **Arithmetic accumulators.** Flagged explicitly by the issue; audited in
  §3/slice 4 — `AddFunction`/`MultiplyFunction` build a fresh accumulator
  and only read `*Arg`; `SubtractFunction` already clones its first argument
  before mutating (`Runtime.cpp:52-53`). None of the three ever mutates
  through a raw `Args` pointer, and those pointers are always into
  `Frame::Done`'s already-materialized `unique_ptr`s (Frames are not
  migrated until S4), so no aliasing reaches the arithmetic layer at all in
  this slice.

## 7. Out of scope

- **S4 ("Value in frames"): `Frame::{Done,Saved,Callee,WcmKeyV}` and
  `MarkFrame` entries moving to `Value`.** This is the slice that will
  actually let a shared lookup flow through to an argument list without
  ever materializing a clone (e.g. genuine zero-copy `(f x x)`). Explicitly
  the next issue per the migration doc's dependency chain; not bundled here.
- **`envExtend` removal or `Environment`'s copy-constructibility.** Dead
  code today, unaffected by this slice's design (map value type is
  unchanged), not touched.
- **Adding `Environment::contains(Id)` (or similar) to replace the
  lookup-and-discard pattern in `envSet`/`isBound` with something that
  never even touches a `shared_ptr` refcount.** A reasonable micro-
  optimization, but not needed for correctness now that `lookup` no longer
  clones; left as a follow-up idea, not bundled into a slice whose point is
  the clone-vs-share behavior change.
- Any GC/Boehm-related work (S9 onward in the migration doc) — no
  collectable value exists yet; not relevant until much later slices.
- Reformatting or renaming unrelated to the `Value`/`Environment` surface.
