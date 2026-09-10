# Plan — Issue #120: M2/GC S4 — Value in continuation frames

## 1. Problem restated

The interpreter's continuation machine (`Interpreter::Frame` and its `Payload`
variant, plus the continuation-mark storage in `ast::MarkFrame`) still stores
runtime values as `std::unique_ptr<ast::ValueNode>`, left over from before the
`Value` handle type landed (S2: registers, S3: environment, #119/PR #195).
Every time a value crosses from the `Val`/`Result` register (already `Value`)
into a Frame field, `Value::takeLegacy()` is called — which *clones* the
value whenever it is currently the `Shared` alternative (an environment
lookup), silently defeating S3's clone-avoidance the moment a looked-up value
is passed as a call argument, saved by `begin0`, or captured by
`with-continuation-mark`. S4 finishes the swap for the continuation machine:
`Frame::{Done,Saved,Callee,WcmKeyV}` and `MarkFrame` entries become `Value`,
so a value can flow from an environment lookup, through application/argument
accumulation, into a saved `begin0` result or a continuation mark, and back
out — without an intermediate clone — while every currently-passing test
(unit + integration, under debug/asan/ubsan) stays green. This is a
**characterization refactor** (doc label CHAR): no new Racket-observable
behavior is intended for the value kinds the interpreter supports today
(`Box`/`Pair`/`Symbol` already carry their own identity through `clone()` via
an internal `shared_ptr<Cell>`/uninterned pointer, so `eq?` cannot regress).
The one thing that *does* change is that primitives (`RuntimeFunction::
operator()`) now sometimes see the live, env-shared `ValueNode` as an
argument instead of a guaranteed-private clone — S4 must re-confirm no
primitive mutates through `Args`, not just add it as a new invariant.

## 2. Files to touch

- `src/include/Interpreter.h` — `Frame::Callee` (header field), `Frame::Seq::
  Saved`, `Frame::App::Done`, `Frame::MkValues::Done`, `Frame::LetBind::Done`,
  `Frame::WcmVal::WcmKeyV` field types; `applyProcedure`'s `Vals` parameter
  type; `bindValues` moves from a free `static` function to a private
  `Interpreter` member (see §3 slice 3 and §5).
- `src/Interpreter.cpp` — `bindValues` body/call sites, `step(Frame::Seq&)`,
  `step(Frame::App&)`, `step(Frame::MkValues&)`, `step(Frame::LetBind&)`,
  `step(Frame::WcmKey&)`, `step(Frame::WcmVal&)`, `applyProcedure`.
- `src/include/ASTRuntime.h` — `MarkEntry` alias (`std::pair<Value, Value>`),
  `setMark`'s declaration; add a direct `#include "Value.h"` (currently only
  transitively visible via `Environment.h`; `misc-include-cleaner` requires a
  direct include on any line that spells `Value` — see
  `clang-tidy-anon-namespace-conflict` memory for the project's diff-scoped
  clang-tidy gotcha).
- `src/ASTRuntime.cpp` — `cloneMarkFrame`, `setMark` bodies; add the same
  direct `#include "Value.h"`.
- `src/Runtime.cpp` — `ContinuationMarkSetFirstFunction::operator()` and
  `ContinuationMarkSetToListFunction::operator()`, which read `MarkEntry::
  {first,second}` directly (`*E.first`, `E.second->clone()` → `.get()`-based
  access, since `Value` has no `operator*`/`operator->`).
- `test/unit/test_interpreter.cpp` — one new characterization test (§3 slice
  6) pinning identity through the `MarkFrame`/`WcmKeyV` path, which nothing
  today exercises.
- No `test/integration/*.rkt` additions: this slice's regression net is the
  existing corpus (named per-slice in §3); no new Racket-observable behavior
  is introduced.

`Value.h` itself is **not modified** — the existing `share()`/`takeLegacy()`/
`get()`/`toShared()` surface is sufficient; this slice is purely about which
storage locations hold a `Value` versus a raw `unique_ptr`.

## 3. TDD slices

Framing: this is a CHAR (behavior-preserving) refactor, not a
new-failing-test-first slice — there is no Racket-level RED to write. Instead,
per field group, "RED" is the **compile break** the type change forces (every
stale `.takeLegacy()`/`unique_ptr<ValueNode>` call site at that field fails to
build until fixed) — the same methodology the migration doc uses for S0's
"RED — link failure." "GREEN" is: builds clean, and the named regression-net
tests below pass, under `debug` first, then `asan`/`ubsan` at the end of the
slice ladder (see §5). Each slice is one commit (this repo has squash merges
disabled, so each must build and pass its net standalone).

**Ordering constraint:** slice 1 must land before slice 4. After slice 4,
`applyProcedure`'s local `Op` becomes a `Value`; there is no `Value →
unique_ptr` implicit conversion, so `Enc.Callee = std::move(Op)` only
compiles if `Frame::Callee` is already `Value`. (The reverse order also
compiles, because `unique_ptr → Value` *is* implicit — but do it in the
stated order so the diff reads as one direction of travel.) Slices 2, 3, 5,
and 6 are independent of each other and of this constraint.

1. **`Frame::Callee` → `Value`.**
   - Change: `Interpreter.h` field declaration only. Both write sites
     (`Interpreter.cpp:549`, `:559`, `Enc.Callee = std::move(Op);` /
     `Kont.back().Callee = std::move(Op);`) keep compiling unchanged today
     (implicit `unique_ptr → Value` conversion); no other call site reads
     `Callee` (it exists purely for RAII: keeping the closure alive while
     `Control` points into its body).
   - While on this line: `Interpreter.h:107`'s comment "so Control, which
     points into its (cloned) lambda body, outlives the call" is already
     slightly stale independent of this slice (closures aren't deep-cloned
     per call in the current model) — fix the wording while touching this
     block. `Interpreter.cpp:549`'s comment "frees the previous activation's
     closure" → "releases" (a `Value` going out of scope isn't necessarily a
     free).
   - Regression net: full unit suite (no `[m2]`/`[tco]` case specifically
     targets `Callee`; it is exercised by every application/tail-call test,
     e.g. "tail-recursive loop computes the correct value", "mutual tail
     recursion is bounded and correct").

2. **`Frame::Seq::Saved` → `Value`.**
   - Change: `Interpreter.h` field type. `Interpreter.cpp:156`
     (`K.Saved = Val.takeLegacy();` → `K.Saved = std::move(Val);`) and
     `:179-182` — collapse
     `std::unique_ptr<ast::ValueNode> R = K.Begin0 ? std::move(K.Saved) :
     Val.takeLegacy(); Kont.pop_back(); deliver(std::move(R));` to
     `Value R = K.Begin0 ? std::move(K.Saved) : std::move(Val); Kont.pop_back();
     deliver(std::move(R));`.
   - **Ordering hazard:** `K` is a reference into `Kont.back().P` (bound by
     `std::visit` in `continueStep`). Every read of `K.Saved` must happen
     **before** `Kont.pop_back()` invalidates it — keep the existing
     "compute into a local, then pop, then deliver" shape; do not inline the
     ternary into the `deliver(...)` call after the pop. This pattern (read
     out of `K` before `pop_back`/before a `pushK` that may reallocate
     `Kont`) applies to every slice below that touches a `step()` function.
   - Regression net: `test/integration/begin0.rkt`.

3. **`bindValues` → private `Interpreter` member taking `Value`;
   `Frame::LetBind::Done` → `std::vector<Value>`.**
   - Why bundled: `bindValues`'s last parameter and `LetBind::Done`'s element
     type must agree, and `bindValues` also has a second call site
     (`step(Frame::LetRec&)`, `Interpreter.cpp:274`) that benefits from the
     same fix for free (`Val.takeLegacy()` → `std::move(Val)`).
   - Why a member, not a free function with a changed signature: this repo's
     `.clang-tidy` enables both `misc-use-anonymous-namespace` and
     `llvm-prefer-static-over-anonymous-namespace`, which contradict each
     other on any *changed* file-scope free function under the diff-scoped
     CI lint job (see the `clang-tidy-anon-namespace-conflict` memory).
     `bindValues`'s signature line changes (parameter type), which makes it
     a changed line and re-triggers the conflict. Converting it to a private
     `Interpreter::bindValues` member sidesteps both checks (member
     functions aren't subject to either rule) and lets it drop its `Diag`
     parameter (already an `Interpreter` member) and use `Interpreter`'s
     other members directly if useful later. Leave the sibling free function
     `formalsAccept` (`Interpreter.cpp:91`) untouched — it is unrelated to
     this slice and unchanged lines stay grandfathered.
   - Inside `bindValues`: replace `dyn_castU<ast::Values>(Val)` (which needs
     a `unique_ptr<From>&` — `Casting.h:10` — and no longer applies once
     `Val` is a `Value`) with `llvm::dyn_cast_or_null<ast::Values>(Val.get())`,
     matching the existing pattern already used in `step(Frame::Define&)`
     (`Interpreter.cpp:305`). The multi-identifier loop
     (`Vars.add(Ids[Idx], dyn_castU<ast::ValueNode>(EPtr));`) is unaffected —
     it clones a fresh AST node from `ValuesExprRange[Idx]`, not from `Val`.
   - `step(Frame::LetBind&)`: `K.Done.push_back(Val.takeLegacy())` →
     `K.Done.push_back(std::move(Val));`; `std::vector<std::unique_ptr<
     ast::ValueNode>> Vals = std::move(K.Done);` → `std::vector<Value> Vals =
     std::move(K.Done);`; the `bindValues(..., std::move(Vals[I]))` call now
     passes a `Value` directly.
   - Regression net: `let-values*.rkt` (base, 1-4), `letrec-values*.rkt`
     (base, 1-4), `error-let-values-arity.rkt`,
     `error-letrec-values-arity.rkt`.

4. **`Frame::App::Done` → `std::vector<Value>`; `applyProcedure`'s `Vals`
   parameter → `std::vector<Value>`.**
   - Depends on slice 1 (see ordering constraint above).
   - `step(Frame::App&)`: same `push_back(std::move(Val))` /
     `std::vector<Value> Vals = std::move(K.Done);` shape as slice 3.
   - `applyProcedure` body (`Interpreter.cpp:413-563`):
     - `std::unique_ptr<ast::ValueNode> Op = std::move(Vals[0]);` →
       `Value Op = std::move(Vals[0]);`. `!Op` (explicit `operator bool`) and
       `Op.get()` (three `dyn_cast`s: `RuntimeFunction`, `Closure`,
       `CaseLambdaClosure`) are unchanged.
     - `Args.push_back(Vals[I].get());` unchanged (`Value::get()` already
       returns the raw pointer `RuntimeFunction::operator()` wants).
     - `CalleeScope->Vars.add(LF[I], std::move(Vals[I + 1]));` (List formal)
       unchanged in shape — `Environment::add` already takes `Value`; this
       stops going through the implicit `unique_ptr → Value` conversion and
       becomes a direct move.
     - `ListRest`/`Identifier` formal cases build a legacy `ast::List` via
       `appendExpr(std::unique_ptr<ValueNode>&&)` — a still-legacy AST
       container this slice does not touch — so these two call sites need
       `Rest->appendExpr(Vals[I + 1].takeLegacy());` /
       `Lst->appendExpr(Vals[I + 1].takeLegacy());`. This *is* a boundary
       materialization (a shared value becomes a private clone when it enters
       a legacy container), matching the doc's Phase 1 pattern ("the legacy
       alternative is simply materialized at the boundary") — not a bug.
     - `Enc.Callee = std::move(Op);` / `Kont.back().Callee = std::move(Op);`
       now move `Value → Value` directly (this is exactly why slice 1 must
       precede this slice).
   - **Primitive-safety re-check (do this before declaring the slice green):**
     after this slice, an identifier argument can reach
     `RuntimeFunction::operator()` as the live, env-shared `ValueNode` rather
     than a guaranteed-fresh clone (previously `Args[I].get()` always pointed
     at a clone materialized by `takeLegacy()`). Grep `src/Runtime.cpp` for
     any primitive that mutates through an `Args[i]` pointer (`const_cast`,
     a non-const method call, in-place field assignment). Confirmed clean by
     inspection during planning: `AddFunction` only reads args into a fresh
     `Sum` accumulator; `SubtractFunction` explicitly clones the first arg
     (`Sub = std::unique_ptr<ast::Integer>(llvm::cast<ast::Integer>(
     I->clone()));`, `Runtime.cpp:52`) before any `-=`; `EqFunction`,
     `ContinuationMarkSet{First,ToList}Function` only read. Re-run this grep
     at implementation time in case primitives were added since planning.
   - Regression net: "tail-recursive loop computes the correct value", "tail
     recursion runs in bounded continuation space", "non-tail recursion still
     grows the continuation", "mutual tail recursion is bounded and correct"
     (unit); `case-lambda*.rkt` (base, 1-4), `error-arity.rkt`, `closure.rkt`
     (integration, covers the `ListRest`/`Identifier` formal paths).

5. **`Frame::MkValues::Done` → `std::vector<Value>`.**
   - `step(Frame::MkValues&)`: same accumulation shape as slices 3/4.
     `if (Vals.size() == 1) { deliver(std::move(Vals[0])); }` — `deliver`
     already takes `Value`, so this drops the implicit conversion too. The
     multi-value case builds `ast::Values` (still a legacy AST container):
     `Exprs.emplace_back(std::move(Vv));` → `Exprs.emplace_back(Vv.takeLegacy());`
     — another expected legacy-boundary materialization, not a regression.
   - Regression net: `values.rkt`, `values-1.rkt`.

6. **`MarkFrame` entries (`MarkEntry`) + `Frame::WcmVal::WcmKeyV` → `Value`.**
   - Bundled because `setMark`'s two value parameters flow directly into
     `MarkEntry`, and `WcmKeyV` is the value `setMark` is eventually called
     with — splitting them would require an awkward intermediate signature.
   - `ASTRuntime.h`: `using MarkEntry = std::pair<Value, Value>;` (was
     `std::pair<std::unique_ptr<ValueNode>, std::unique_ptr<ValueNode>>`);
     `void setMark(MarkFrame &Frame, Value Key, Value Val);`. Add
     `#include "Value.h"`.
   - `ASTRuntime.cpp`:
     - `cloneMarkFrame`: `Out.emplace_back(std::unique_ptr<ValueNode>(
       E.first.get()->clone()), std::unique_ptr<ValueNode>(
       E.second.get()->clone()));` (the `unique_ptr → Value` conversion
       happens implicitly as `pair`'s converting constructor forwards to
       `Value`'s converting constructor — no explicit `Value(...)` wrapper
       needed, matching the existing `std::unique_ptr<ast::ValueNode>(X.clone())`
       idiom used throughout `Interpreter.cpp`).
     - `setMark`: `valueEq(*E.first, *Key)` → `valueEq(*E.first.get(),
       *Key.get())` (no `operator*`/`operator->` on `Value`); body otherwise
       unchanged (`E.second = std::move(Val);` / `Frame.emplace_back(
       std::move(Key), std::move(Val));` already work on `Value`s).
   - `Interpreter.h`: `Frame::WcmVal::WcmKeyV` field type → `Value`.
   - `Interpreter.cpp`:
     - `step(Frame::WcmKey&)`: `std::unique_ptr<ast::ValueNode> KeyV =
       Val.takeLegacy(); ... WV.WcmKeyV = std::move(KeyV);` collapses to
       `WV.WcmKeyV = std::move(Val);` (drop the now-pointless intermediate).
     - `step(Frame::WcmVal&)`: `std::unique_ptr<ast::ValueNode> KeyV =
       std::move(K.WcmKeyV);` → `Value KeyV = std::move(K.WcmKeyV);` — read
       out of `K` **before** the subsequent `Kont.pop_back()` (same hazard as
       slice 2). `std::unique_ptr<ast::ValueNode> ValV = Val.takeLegacy();`
       → `Value ValV = std::move(Val);`. The two `ast::setMark(Kont.back().
       Marks, std::move(KeyV), std::move(ValV));` calls are unchanged in
       shape (now pass `Value`s directly, matching the new signature).
   - `Runtime.cpp`: in both `ContinuationMarkSetFirstFunction::operator()`
     and `ContinuationMarkSetToListFunction::operator()`, `valueEq(*E.first,
     *Key)` → `valueEq(*E.first.get(), *Key)` and `E.second->clone()` →
     `E.second.get()->clone()`.
   - **New characterization test** (`test/unit/test_interpreter.cpp`, near
     the existing WCM tests at lines 303-335): nothing today exercises
     identity *through* a continuation mark — `(eq? b b)` exercises
     `App::Done`, but `Box`/`Pair` already preserve identity across `clone()`
     via their internal cell regardless of this slice, so it doesn't pin the
     `MarkFrame` path specifically. Add: bind a box, install it as a mark via
     `with-continuation-mark`, retrieve it via
     `continuation-mark-set-first`/`current-continuation-marks`, `set-box!`
     through the retrieved reference, and confirm the mutation is visible
     through the original `b` (mirrors the existing "mutate through one
     reference, observe through another" template from S3, per the doc's
     standing guard, R5). This test must pass both before and after this
     slice (Box's own identity mechanism already guarantees it) — its
     purpose is to lock in the invariant going forward, not to prove new
     behavior.
   - Regression net: `with-continuation-mark.rkt` through `...7.rkt`,
     `wcm-var-key.rkt` (integration); "a tail call through
     with-continuation-mark runs in bounded space", "a tail-position
     with-continuation-mark replaces, not accumulates, a same-key mark across
     loop iterations" (unit, `test_interpreter.cpp:303-335`) — these already
     exercise exactly this code path and are the strongest existing net for
     this slice; plus the new box/mark test above.

## 4. AST/visitor surface

None. This slice changes field *storage types* inside `Interpreter::Frame`
(a private, non-`ASTNode` implementation-detail struct) and inside
`ast::MarkFrame`/`MarkEntry` (a `using` alias, not an `ASTNodeKind`). No
`ASTNode` kind is added, removed, or reordered; no visitor (`ASTVisitor`,
`Interpreter`) gains or loses a `visit()` overload. `ast::ContinuationMarkSet`
itself (the `ASTNode`/`ValueNode` that reifies a mark snapshot) is unchanged
by this slice — only what it stores per-frame changes type; its GC-cell
migration is S15, out of scope here.

## 5. Risk areas

- **Primitive mutation through shared `Args`** (slice 4's primary risk,
  detailed above) — the one place this CHAR slice could silently become a
  correctness bug instead of a pure refactor. Re-verify at implementation
  time, not just at planning time, in case new `RuntimeFunction`s landed on
  `main` since this plan was written.
- **Reference invalidation across `pop_back()`/`pushK()`** (slices 2 and 6) —
  `K` in every `step(Frame::X&)` is a reference into `Kont.back().P`; moving
  out of `K`'s fields must happen before any operation that can invalidate
  `Kont`'s backing storage (`pop_back`, or `pushK`'s `emplace_back`, which can
  reallocate). The existing code already gets this right (computes into a
  local before popping); the risk is a "simplification" during this refactor
  that inlines a `K.Field` read after the pop. Guard: run `asan` (catches
  use-after-free on the `Frame` itself, though not necessarily a stale
  reference read from a moved-from `Value` — review each `step()` diff by
  eye against this rule, don't rely on the sanitizer alone).
- **`write()`/Racket-printed-output risk: none.** This slice touches no
  `write()` override and no value's printed representation; the `FileCheck`
  `CHECK:` lines in the regression-net `.rkt` files assert on final
  `ast::ValueNode::write()` output, which is unaffected by how a value was
  stored in-flight.
- **Free-variable analysis risk: none.** `AnalysisFreeVars` operates on the
  AST (`ExprNode`/`Identifier`), not on `Interpreter::Frame` or `MarkFrame`.
- **Memory safety / lifetime:** no new ownership model is introduced — this
  slice only changes which of `Value`'s two existing alternatives
  (`Legacy`/`Shared`) a storage location can hold, using `Value`'s existing,
  already-reviewed move semantics. Still, because `Value` is move-only and
  several fields move to `std::vector<Value>`, run the full suite under
  `asan` and `ubsan` presets at the end of the slice ladder (not just
  `debug` per-slice) to catch any move-related double-use the compiler
  doesn't.
- **clang-tidy diff-scoped lint conflict** (slice 3) — see the
  `clang-tidy-anon-namespace-conflict` memory; addressed by making
  `bindValues` a member function rather than adjusting its free-function
  form, which would still trip one of the two contradictory checks.
- **`misc-include-cleaner`** — every new line spelling `Value` in
  `ASTRuntime.h`/`ASTRuntime.cpp` needs the direct `#include "Value.h"` even
  though it's already transitively reachable via `Environment.h`; verify
  `Runtime.cpp`'s new `.get()` call sites don't need one too (likely fine —
  they call a method on an already-deduced `Value`, not naming the type).
- **Verification commands per slice:** `cmake --build --preset debug -j6`
  then `ctest --preset debug` (per `nora-toolchain-versions` memory: cap
  parallelism at `-j6`, shared 32 GB cgroup). Format with
  `/usr/bin/clang-format` explicitly (not the PATH-shadowing v11 at
  `~/.local/bin/clang-format`). After all six slices are green under
  `debug`: `cmake --build --preset asan -j4` / `ctest --preset asan`, then
  the same for `ubsan` (lower `-j` for the heavier sanitizer builds under the
  shared memory cap per the run's operating contract).

## 6. Out of scope

- **The remaining `Val.takeLegacy()` call sites that are not Frame fields**
  in scope for #120: `step(Frame::IfBranch&)` (`Interpreter.cpp:190`),
  `step(Frame::Define&)` (`:297`), `step(Frame::Set&)` (`:332`),
  `step(Frame::WcmMark&)` (`:391`), `step(Frame::Call&)` (`:399`), and
  `visit(Linklet const&)`'s `Last = Val.takeLegacy();` (`:121`). These still
  materialize a clone whenever `Val` is `Shared`, but none of them touch a
  field named in the issue (`Done`/`Saved`/`Callee`/`WcmKeyV`) or
  `MarkFrame`. Leave them for a later slice (`IfBranch`'s is naturally
  revisited when S6/booleans lands per the doc's ladder; the rest are
  candidates for the eventual S18 seam collapse). Do not fold them into this
  PR.
- **`ContinuationMarkSet`/`MarkFrame` becoming actual GC cells** — that is
  S15 in the migration doc, much later in Phase 4; this slice only changes
  what *type* a mark's key/value pair holds while `MarkFrame` itself remains
  a plain `std::vector` in interpreter-owned (not GC) storage.
  `Frame::Callee` similarly stays a `Value` handle, not a GC pointer, for
  the whole of this slice.
- **No `clang-tidy`/formatting-only cleanups** beyond what each slice's
  touched lines force, and the two comment fixes named in slice 1 (which are
  directly adjacent to lines this slice already rewrites, not drive-by
  cleanup elsewhere in the file).
- **No new `RuntimeFunction`s or Racket-level features.** This is purely an
  internal storage-representation change.
- **`expander/expander.rktl`** — not touched, not relevant to this slice.
- **MLIR (`src/mlir/`, `src/include/nir/`)** — not touched, not relevant.
