# Architecture review — nora — 2026-09-11

**Scope**: The M2/GC immediate-migration hot spot — `src/Interpreter.cpp`,
`src/include/Value.h`, `src/Runtime.cpp`, `src/include/AST.h`, and the runtime
supporting files. Chosen because the last stretch of `git log` (S5–S7 immediates
#197/#199/#201, the `toShared()` fix #198, the builtin-registry seam #200) keeps
landing there; deepening pays off where change concentrates.
**Picked**: `value-register-take-vs-borrow` — see PR #202 and `.architecture/backlog.md`.
**Degradations**: advisor rate-limited only at the step-3 approach-validation
call; it was available at the step-4 design adjudication and confirmed the pick.
Parallel sub-agents were used for both exploration and design-it-twice.

> **Diagram legend**: solid edges are the module's public interface; dashed
> edges are calls that live *inside* the implementation, behind the seam.

## Candidates

### value-register-take-vs-borrow — give the `Value` register a borrow seam · Strong · score 22/25

- **Files**: `src/include/Value.h`, `src/Interpreter.cpp:198,306,341,400,408`
  (the five avoidable sites); file-count estimate **~2**.
- **Score**: 22/25
  - **Leverage 4** — five `step`/apply arms stop reaching for the consume-or-clone
    path; the whole register-handoff idiom in the hot interpreter loop simplifies,
    and the avoidable clones #195/#196 removed stop creeping back.
  - **Locality 4** — the take-vs-peek decision moves *onto the handle* instead of
    being re-made, representation-aware, at every call site.
  - **Blast radius 1** — one header seam plus five one-line call-site edits, no
    published interface; ~2 files.
  - **Heat 5** — `Value.h` (5×) and `Interpreter.cpp` (9×) are the hottest files
    in the recent window; this is exactly the mid-migration code S8 will touch next.
- **Problem** — `Value` carries three representations (`Legacy` unique_ptr,
  `Shared` shared_ptr, `Imm` immediate word) and exposes three consuming/peeking
  reads: `get()` (peek, non-consuming, non-cloning), `takeLegacy()`
  (consume, **and clone if `Shared`**), `toShared()` (consume into a binding).
  The *cost* of `takeLegacy()` depends on which representation is engaged, and
  that leaks: five interpreter arms reflexively call `takeLegacy()` where they
  only need to **borrow** (read through the handle) or to **move the whole handle
  into a sink that already takes a `Value`**. Each such call clones a shared
  value for nothing — re-introducing exactly the copies #195/#196 removed.
- **Deletion test** — **PASS**. A borrow seam on `Value` concentrates the
  peek-vs-consume choice behind the handle. Delete it and the choice re-smears
  across the arms, each re-deciding — representation-aware — whether to clone.
  Complexity concentrates, it does not merely move.
- **Solution** — Recognise (and make explicit at the interface) that `Value`
  already sinks by move — `deliver(Value)`, `envSet(…,Value)`,
  `Environment::add(…,Value)`, `toShared()` all take a `Value` — and that `get()`
  is the non-cloning borrow. Add a small, intent-named borrow accessor so the
  read-only arms stop consuming, then route the five avoidable sites:
  - `IfBranch` fallback (198): borrow instead of `takeLegacy()` to read the bool.
  - `WcmMark` (400) / `Call` (408): `deliver(std::move(Val))` — pass the handle
    straight through, no materialise-and-clone.
  - `Set` (341): `envSet(E, *Id, std::move(Val))` — the sink takes a `Value`.
  - `Define` single-id (306): `add(id, std::move(Val))`; multi-id inspection
    borrows rather than consuming.
  `takeLegacy()` stays only for the three **genuine** into-`unique_ptr<ExprNode>`
  sinks (`MkValues` 237, `applyProcedure` rest/id children 525/534), which are
  rooted in AST nodes storing `unique_ptr` children (the value-in-AST conflation)
  — explicitly out of scope here.
- **Benefits** — *leverage*: the register handoff in the hottest loop becomes
  uniform (move to sink, borrow to read); *locality*: "does this clone?" is
  answered once, by the handle, not per arm; *test surface*: the borrow/no-clone
  invariant becomes unit-pinnable — build two `Value`s over one `shared_ptr`,
  borrow/move one, and assert pointer identity is preserved (no clone) — which is
  impossible to state against the current representation-leaking surface.

```mermaid
graph LR
  A1[IfBranch arm] --> TL[takeLegacy consume+clone]
  A2[Set arm] --> TL
  A3[Define arm] --> TL
  A4[WcmMark arm] --> TL
  A5[Call arm] --> TL
  TL --> R[(Shared / Legacy / Imm)]
```

```mermaid
graph LR
  A1[IfBranch arm] --> B[borrow peek]
  A2[Set arm] --> S[move into sink]
  A3[Define arm] --> S
  A4[WcmMark arm] --> S
  A5[Call arm] --> S
  B -.-> R[(Shared / Legacy / Imm)]
  S -.-> R
```

### formal-deep-interface — give the `Formal` tag a deep interface · Worth exploring · score 21/25

- **Files**: `src/include/AST.h`, `src/AST.cpp:238-266`, `src/Interpreter.cpp:91-99,463-539`,
  `src/AnalysisFreeVars.cpp:43-52`; estimate ~3.
- **Score**: 21/25 — **leverage 4** (five `Formal::Type` re-dispatch sites collapse;
  fixes the per-application `auto` copy of the identifier `SmallVector`);
  **locality 4** (arity/binding/dump own their behaviour once per subclass);
  **blast radius 1** (~3 files, no published surface); **heat 4**.
- **Problem** — `Formal` exposes only its `getType()` tag; arity (`formalsAccept`),
  the duplicated closure-arity check in `applyProcedure`, argument binding,
  `Lambda::dump`, and the (dead) `AnalysisFreeVars` all switch on the tag and
  re-`static_cast`. The interface is a tag; the behaviour lives in the callers —
  shallow. `Lambda::getFormalsType()` (AST.h:729) is a tag-leak accessor with
  no production callers (5 test sites only).
- **Deletion test** — **PASS**. `accepts(nargs)` / `bind(...)` / `boundVars()` /
  `dump()` virtuals concentrate five re-dispatches into one per-subclass body.
- **Solution** — Move the per-`Type` behaviour behind virtuals on `Formal`;
  callers ask `F.accepts(n)` / `F.bind(...)`. Drop the callerless `getFormalsType`.
- **Benefits** — leverage across five sites; the switch-and-cast idiom (and its
  `auto`-copy footgun) disappears; adding a new formal kind stops touching five
  files. Test surface: each formal kind's arity/binding testable in isolation.

```mermaid
graph LR
  C1[formalsAccept] --> T{Formal::Type}
  C2[applyProcedure arity] --> T
  C3[arg-binding] --> T
  C4[Lambda::dump] --> T
```

```mermaid
graph LR
  C1[formalsAccept] --> F[Formal.accepts/bind]
  C2[applyProcedure arity] --> F
  C3[arg-binding] --> F
  C4[Lambda::dump] --> F
  F -.-> L[ListFormal]
  F -.-> LR2[ListRestFormal]
  F -.-> I[IdentifierFormal]
```

### clone-unique-wrap — one owning-clone seam instead of 26 wrap-sites · Worth exploring · score 19/25 *(new)*

- **Files**: `src/include/AST.h` (the `ClonableNode` template, :132-146), and 26
  wrap-sites across `Interpreter.cpp`, `Runtime.cpp`, `ASTRuntime.cpp`, `AST.cpp`;
  estimate ~5.
- **Score**: 19/25 — **leverage 4** (26 identical `unique_ptr<T>(x->clone())`
  wraps collapse; a leak-prone ceremony is centralised); **locality 3**;
  **blast radius 2** (~5 files, mechanical, no published surface); **heat 4**.
- **Problem** — `ASTNode::clone()` returns a raw owning `ASTNode*` (covariant
  `ValueNode*`), and every one of 26 callers re-wraps it into a `unique_ptr`.
  The interface hands out a raw owning pointer; each caller performs the same
  ownership ceremony. `Formal::clone()` already returns `unique_ptr<Formal>` —
  the deep version to mirror.
- **Deletion test** — **PASS**. A `cloneUnique(node)` / non-virtual `clonePtr()`
  owns the wrap once; deleting it scatters the ceremony back to 26 sites.
- **Solution** — Add a non-virtual `clonePtr()` on `ClonableNode` (or a free
  `cloneUnique`), keeping the virtual's covariant raw return (unique_ptr is not
  covariant, so the virtual must stay raw). Route the 26 sites through it.
- **Benefits** — leverage across 26 sites; ownership handled once; the raw
  `clone()` stops being the thing callers reach for. Test surface: unchanged
  behaviour, pinned by existing tests.

```mermaid
graph LR
  K1[caller x26] --> RC[raw clone ASTNode*]
  K1 --> W[manual unique_ptr wrap]
```

```mermaid
graph LR
  K1[caller x26] --> CU[cloneUnique]
  CU -.-> RC[raw clone]
  CU -.-> W[wrap once]
```

### range-adapter-dedup — one generic range instead of five hand-rolled adapters · Worth exploring · score 19/25 *(new)*

- **Files**: `src/include/AST.h` (`DefineValues::IdRange` :536-547, `ListFormal::IdRange`
  :620-630, `LetValues::IdRange` :787-800, `Linklet::FormRange` :282-294,
  `Values::ExprRange` :934); estimate ~1-2.
- **Score**: 19/25 — **leverage 3** (three byte-identical `IdRange` classes and
  two sibling adapters collapse to one generic; the code's own FIXMEs at :534 and
  :784 ask for it); **locality 4**; **blast radius 1** (1-2 files, all `AST.h`);
  **heat 4**.
- **Problem** — Five hand-rolled range adapters, three of them byte-identical,
  each with an interface (`begin`/`end`/`operator[]`) as large as its body — the
  textbook shallow module. The FIXMEs admit a `view_interface` was intended.
- **Deletion test** — **PASS**. One `IterRange<T>` (or `std::ranges::subrange`)
  concentrates the adapter; the five copies were pure boilerplate.
- **Solution** — Replace the five with one generic range type / `subrange`.
- **Benefits** — deletes duplicated boilerplate; one place to fix a range bug.
  Test surface: existing range-for iteration tests cover it.

```mermaid
graph LR
  N1[DefineValues] --> IR1[IdRange copy 1]
  N2[ListFormal] --> IR2[IdRange copy 2]
  N3[LetValues] --> IR3[IdRange copy 3]
```

```mermaid
graph LR
  N1[DefineValues] --> G[IterRange<T>]
  N2[ListFormal] --> G
  N3[LetValues] --> G
```

### parse-form-combinators — fold the form prologue into combinators · Worth exploring · score 20/25

- **Files**: `src/Parse.cpp`, `src/include/Parse.h`; estimate ~2 (~18 form parsers).
- **Score**: 20/25 — **leverage 4**, **locality 4**, **blast radius 2**, **heat 4**.
- **Problem** — Every form parser repeats a `getPosition`/`gettok`/LPAREN-guard/
  `rewindTo` prologue (22 LPAREN guards, 55 `rewindTo`, 18 emit-once guards) and
  the `if (!hadError) parseError` idiom. The save/rewind/emit-once invariant is
  re-implemented per parser.
- **Deletion test** — **PASS**. `openForm(S, keyword)` / `expect(S, tok, msg)`
  combinators own the invariant once.
- **Solution** — Extract the combinators; a keyword→parser table for `parseExpr`.
- **Benefits** — the invariant lives once; each parser shrinks to its shape.
  Test surface: existing parse unit tests plus new combinator-level tests.

```mermaid
graph LR
  P1[parseLambda] --> PR[save/gettok/LPAREN/rewind prologue]
  P2[parseCaseLambda] --> PR
  P3[parseLet...] --> PR
```

```mermaid
graph LR
  P1[parseLambda] --> OF[openForm/expect]
  P2[parseCaseLambda] --> OF
  P3[parseLet...] --> OF
  OF -.-> PR[save/gettok/rewind/emit-once]
```

### continuation-mark-query-seam — a query seam on `ContinuationMarkSet` · Worth exploring · score 19/25

- **Files**: `src/Runtime.cpp:82-89,105-113`, `src/include/ASTRuntime.h:164`,
  `src/ASTRuntime.cpp`; estimate ~3.
- **Score**: 19/25 — **leverage 3**, **locality 4**, **blast radius 2**, **heat 5**.
- **Problem** — `ContinuationMarkSet` exposes only `getFrames()`;
  `continuation-mark-set-first` and `continuation-mark-set->list` both reach past
  it and re-implement the same innermost-first key scan.
- **Deletion test** — **PASS**. `firstForKey(key)` / `allForKey(key)` own the scan.
- **Solution** — Add the two query methods; both builtins call them.
  (The bundled `valueEq`-by-name vs `eq?`-by-identity drift is a correctness
  concern excluded from this pure seam extraction — left for a human.)
- **Benefits** — the scan lives once; the builtins shrink to a query call.
  Test surface: the scan becomes unit-testable on `ContinuationMarkSet` directly.

```mermaid
graph LR
  B1[cms-first] --> GF[getFrames + manual scan]
  B2[cms->list] --> GF
```

```mermaid
graph LR
  B1[cms-first] --> Q[firstForKey/allForKey]
  B2[cms->list] --> Q
  Q -.-> GF[getFrames scan]
```

### bind-result-helper — one destructuring helper for let/letrec/define · Worth exploring · score 19/25

- **Files**: `src/Interpreter.cpp:60-87` (`bindValues`) and `:303-336` (inline
  `Frame::Define` arm); estimate ~1.
- **Score**: 19/25 — **leverage 3**, **locality 4**, **blast radius 1**, **heat 4**.
- **Problem** — `bindValues` and the inline `Define` arm both do 1-id/N-id
  multiple-values destructuring, duplicated with divergent error text and (now)
  divergent `Value`-seam handling (overlaps `value-register-take-vs-borrow`).
- **Deletion test** — **PASS**. One `bindResult` helper owns the arity/destructure/
  error contract.
- **Solution** — Extract `bindResult`; route both arms through it. Best sequenced
  *after* the `Value`-seam pick so the two arms share the same borrow handling.
- **Benefits** — one destructuring contract; consistent error text.
  Test surface: destructuring arity errors testable once.

```mermaid
graph LR
  L[let/letrec] --> BV[bindValues]
  D[define arm] --> DUP[inline duplicate]
```

```mermaid
graph LR
  L[let/letrec] --> BR[bindResult]
  D[define arm] --> BR
```

### environment-deepen — one scope module owning teardown · Worth exploring · score 19/25

- **Files**: `src/include/Environment.h`, `src/Environment.cpp`, and the
  interpreter-owned `AllScopes` half in `src/include/Interpreter.h` /
  `src/Interpreter.cpp:40-58`; estimate ~4.
- **Score**: 19/25 — **leverage 4**, **locality 4**, **blast radius 2**, **heat 3**.
- **Problem** — Understanding one binding means bouncing across `Environment`,
  `Scope`, free functions (`envExtend`/`envLookup`/`envSet`), and the
  interpreter-owned cycle-breaking. `envSet` fabricates a whole `Value` (a
  discarded refcount bump) just to test presence; a `contains()` predicate is
  missing. Dead surface: `envExtend`, `Environment::begin/end` have zero callers.
- **Deletion test** — **borderline-PASS**. The win depends on an arena/scope
  module absorbing the `AllScopes` teardown plus `contains()`, not merely moving
  free functions onto methods.
- **Solution** — Fold the scope chain + teardown into one module with
  `contains()`, arena ownership, and pointer-identity keys; delete the dead surface.
- **Benefits** — one place to understand binding lifetime; presence test without
  fabricating a `Value`. Test surface: teardown/cycle-breaking becomes testable
  without an `Interpreter`.

```mermaid
graph LR
  I[Interpreter] --> AS[AllScopes teardown]
  I --> EF[envExtend/Lookup/Set]
  EF --> SC[Scope]
  SC --> EN[Environment]
```

```mermaid
graph LR
  I[Interpreter] --> SM[Scope module]
  SM -.-> AS[teardown]
  SM -.-> EN[Environment + contains]
```

### visitor-defaults-dead-code — default the visitor, delete the dead pass · Speculative · score 16/25

- **Files**: `src/include/ASTVisitor.h:11-39`, `src/AnalysisFreeVars.{cpp,h}`,
  `src/Interpreter.cpp:761-825`, CMake; estimate ~3-5.
- **Score**: 16/25 — **leverage 3**, **locality 3**, **blast radius 2**, **heat 3**.
- **Problem** — 29 pure-virtual `visit` methods force every visitor to override
  all; `AnalysisFreeVars` is entirely dead (and its `visit(Identifier)` is
  latently buggy); `Lambda::findFreeVariables` is declared, never defined;
  14 byte-identical `deliver(clone())` self-quoting overrides remain (was 17;
  #199/#201 immediates removed 3).
- **Deletion test** — **PASS on the dead half** (delete `AnalysisFreeVars` +
  `findFreeVariables`); the no-op-defaults + a `visitSelfQuoting` hook is a
  genuine but low-leverage narrowing.
- **Solution** — Default the visitor's pure virtuals to no-ops; delete the dead
  pass and undefined declaration; optionally collapse the self-quoting overrides.
- **Benefits** — less required boilerplate per visitor; dead code gone.
  Test surface: unchanged.

```mermaid
graph LR
  V1[Interpreter] --> PV[29 pure virtuals]
  V2[AnalysisFreeVars dead] --> PV
```

```mermaid
graph LR
  V1[Interpreter] --> DV[defaulted visitor]
  DV -.-> NO[no-op defaults]
```

### operand-accumulate-seam — one advance-or-finish seam across step arms · Speculative · score 18/25 *(new)*

- **Files**: `src/Interpreter.cpp:207-251` (`step(App)`, `step(MkValues)`,
  `step(LetBind)`); estimate ~1.
- **Score**: 18/25 — **leverage 3**, **locality 2**, **blast radius 1**, **heat 5**.
- **Problem** — `step(App)`, `step(MkValues)`, and `step(LetBind)` share a
  byte-identical accumulate-then-advance prologue; only the "all done" epilogue
  differs.
- **Deletion test** — **PASS (modest)**. An `advanceOrFinish(K)` seam owns the
  operand state machine.
- **Solution** — Extract the accumulate/advance prologue; keep the per-arm epilogue.
- **Benefits** — the operand loop lives once. Test surface: unchanged.

```mermaid
graph LR
  S1[App] --> PRO[push/advance prologue]
  S2[MkValues] --> PRO
  S3[LetBind] --> PRO
```

```mermaid
graph LR
  S1[App] --> AF[advanceOrFinish]
  S2[MkValues] --> AF
  S3[LetBind] --> AF
```

## Dropped

| Candidate | Dropped because |
|---|---|
| `abort-eval-fail-helper` (`Diag.error; abortEval; return` epilogue ×11) | Leverage 2 — a 3-line epilogue collapse; interface barely deepens, callers do the same work. Recorded for the next run, not picked. |

## Too large to automate

None this run — no candidate scored blast radius 5.

## Pick

**`value-register-take-vs-borrow` (22/25)** is taken. It is the top surviving
`proposed` candidate after `runtime-builtin-boilerplate` (#200) merged, and it
was already recorded as the prior run's "natural next firing." The runner-up
**candidate** is `formal-deep-interface` (21/25) — **within 1 point**, so this
pick was close and `formal-deep-interface` is the natural next firing. The pick
wins on heat (5 vs 4): it sits in the exact mid-migration `Value`/interpreter
code the next slice (S8 fixnums) will touch, so the borrow seam pays off soonest,
and its blast radius (~2 files, one header seam + five one-line edits) is the
smallest of the top tier.

New candidates `clone-unique-wrap` and `range-adapter-dedup` (both 19/25) score
below the pick and are recorded in the backlog for future firings.

## Design

Three designs were produced by parallel sub-agents, each with a radically
different mandate. The adjudication was confirmed by the advisor (available at
step 4; it was only rate-limited at the step-3 approach check — see Degradations).

### Design A — minimal surface (no interface change)

Add nothing to `Value`. Recognise `get()` as the borrow and "move the whole
`Value` into a sink that already takes one" as the into-sink path; fix the five
sites with `get()` + `std::move(Val)`. Net −1 line, one file.
**Rejected.** It does not deepen the module at all — `Value` stays exactly as
shallow, and `takeLegacy()` remains the easy footgun. Decisively, under the
test-first rule it **cannot produce a red test**: its proposed unit test pins the
`get()`/move contracts that *already hold*, so it is green before the change.
A design whose pinning test never goes red fails step 5 by construction.

### Design B — intent-named typed borrow `as<T>()`  *(WINNER)*

Add one non-consuming, non-cloning typed peek to `Value`:

```cpp
template <std::derived_from<ast::ValueNode> T>
[[nodiscard]] const T *as() const { return llvm::dyn_cast_or_null<T>(get()); }
```

Read-only arms ask `V.as<ast::BooleanLiteral>()` / `V.as<ast::Values>()` /
`Op.as<ast::Closure>()`; pass-through arms move the whole `Value` into their
sink; `takeLegacy()` is left only for the three genuine `unique_ptr<ExprNode>`
AST-child sinks, becoming visibly transfer-only. The `std::derived_from`
constraint makes `as<ast::Identifier>()` (a non-`ValueNode`) a compile error
rather than an always-null silent bug. Adds `#include <llvm/Support/Casting.h>`
to `Value.h` (`AST.h` does not transitively provide it).

### Design C — consume-policy split (rename `takeLegacy` → `cloneOutLegacy`)  *(runner-up design)*

Rename the consuming method so the clone cost is in the name; reroute the five
avoidable sites to borrow/move and rename the three genuine sinks to
`cloneOutLegacy()`. Makes an accidental clone unrepresentable (you must type
"clone"). Real improvement, but it adds **naming, not behaviour**, has the
largest blast radius (4 files, plus a prose reference in `ASTRuntime.h:67` and
two renamed test titles), and renames a method the migration schedules for
deletion at S18.

### Adjudication

Criteria in order — depth, locality, seam placement, test surface, blast radius:

1. **Depth** — B adds a genuinely deep verb (materialize-if-immediate +
   non-cloning-borrow + typed-downcast, behind one call); A adds nothing;
   C only renames. **B.**
2. **Locality** — B concentrates "peek register as AST kind T" behind the handle
   (used at ≥6 arms); C concentrates the clone decision in a named method; A
   leaves each arm open-coding. **B ≥ C > A.**
3. **Seam placement** — "peek as type T" varies at ≥2 real sites
   (`IfBranch`/`Define`/`bindValues`/operator dispatch), so the seam is real, not
   hypothetical; B places it there. **B.**
4. **Test surface** — B and C both pin non-consuming/non-cloning through the
   interface (pointer identity vs the backing `shared_ptr`); A admits its test
   cannot pin the call sites. **B ≈ C > A.**
5. **Blast radius** — A smallest, B medium (3 files), C largest — a tiebreak only,
   and the first four criteria already separate the designs.

**Winner: Design B.** It is the only design that deepens the interface (the whole
point of the exercise) while keeping the diff contained.

### Scope decided (recorded so a reviewer sees it was chosen, not missed)

- **Route through `as<T>()`** the six read-only peeks: `bindValues` (67),
  `IfBranch` (198), `Define` multi-id inspection (314), and — for one spelling of
  "peek register as type" — the operator dispatch in `applyProcedure`
  (432/461/484). All operator accessors (`getName`/`getLambda`/`getEnv`/
  `getCaseLambda`) are `const`, so `as<T>()` returning `const T*` is safe there.
- **Move the whole `Value` into the sink** at the four pass-through/store arms:
  `Define` single-id (306) → `add(id, std::move(Val))`, `Set` (341) →
  `envSet(…, std::move(Val))`, `WcmMark` (400) / `Call` (408) →
  `deliver(std::move(Val))`.
- **Keep `takeLegacy()`** at the three genuine `unique_ptr<ExprNode>` AST-child
  sinks (`MkValues` 237, `applyProcedure` rest/id 525/534); update its doc to
  name them as the transfer-only survivors.

### Behaviour-preservation argument

On the `Shared` path, `Set` and `Define`-single-id change from clone-then-store
to move-the-`shared_ptr`, so a binding now **aliases** the source node instead of
holding a private copy. This is unobservable: `Box`/`Pair` keep their mutable
state and `identity()` in a **shared cell that survives `clone()`**, and `eq?`
compares those cell pointers, not `ValueNode` pointers — so clone and alias are
observationally identical for the only identity/mutation-sensitive types, and the
alias is in fact the correct Racket sharing semantics. All three sub-agents
reached this independently. Confirmation is empirical: the **full integration
suite** (box/pair/set!/eq? coverage) must pass, not just the unit tests.

### Test-pinning note (per the immediacy-assertion check)

No existing test pins the *old* materialization as a requirement. The
`RawImmediate` assertions (`test_interpreter.cpp` 396–453) are all *positive* and
either top-level or on paths this change does not touch; test 406 pins the
`IfBranch` *fallback* (preserved — `as<>` still walks it); and the `set!` test
(457) explicitly declines to pin `RawImmediate`, anticipating exactly this
Call/WcmMark deferral. So no test is weakened to reach green. After the change an
immediate may survive a `Call`/`WcmMark` frame pop *as an immediate* (a saved
allocation), which only ever satisfies more `expect*`/`RawImmediate` assertions,
never contradicts one.

### Blast radius (actual)

3 files: `src/include/Value.h` (+1 include, +~11-line `as<T>()`, doc touch),
`src/Interpreter.cpp` (~10 arms edited, net negative), and
`test/unit/test_environment.cpp` (+`as<T>()` unit tests). The candidate was
scored at ~2 files (blast radius 1); actual 3 is within the step-5 2× guard.
