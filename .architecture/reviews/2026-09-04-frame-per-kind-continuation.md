# Architecture review — nora — 2026-09-04

**Scope**: The interpreter (`src/`), weighted to the git hot-spots of recent history — `Interpreter.cpp`, `AST.h`/`AST.cpp`, `Interpreter.h`, `Parse.cpp` are the files that keep changing (proper tail calls, the M2/GC value-model migration, the value-printing seam). Cold code (`Environment.cpp`, the MLIR scaffold) is scored but docked by YAGNI. This is the third firing; it reconciles against a persisted backlog of six candidates and the one that already landed.

**Picked**: `frame-per-kind-continuation` — see PR #191 and `.architecture/backlog.md`.

**Degradations**: none. `gh` authenticated; sub-agent exploration and design-it-twice both available; `codebase-design` vocabulary in use.

**Diagram convention** (replaces the upstream HTML legend): in every Mermaid block, **solid edges are the module interface** a caller or the machine driver sees; **dashed edges are inside the implementation**, hidden behind the seam.

---

## Candidates

### frame-per-kind-continuation — one small continuation type per Kind, with `resume()` · Strong · score 22/25

- **Files** — `src/include/Interpreter.h:100` (the `Frame` struct), `src/Interpreter.cpp:143` (`continueStep`, the 13-arm switch), and the 14 push-sites at `Interpreter.cpp:635,647,667,684,711,731,740,750,758` (Eval-mode `visit`s) plus `:109,355,391,569` (driver/internal pushes). File-count estimate: **~3** (`Interpreter.h`, `Interpreter.cpp`, `test/unit/test_interpreter.cpp`).
- **Score — 22/25**
  - *Leverage 5* — the deepening pays back across the whole CEK loop, not one call site: every one of the 13 continuation kinds gains a home for its own fields and its own resume behaviour, and adding a future kind (delimited-continuation prompts, `call/cc` — issue #11) stops touching a shared struct **and** a shared switch **and** scattered field declarations.
  - *Locality 5* — today a continuation kind's data (`Interpreter.h`) is one place, its construction (the `visit` push-site) a second, and its transition (the `continueStep` arm) a third. After: all three sit on one type. A change to how, say, `LetBind` resumes becomes a one-type edit.
  - *Blast radius 3* — only two source files, no published interface (the `Frame` type is entirely private to `Interpreter`; `main.cpp` never sees it), but ~470 lines move and the change crosses the **Boehm-GC seam** (`Kont` is `std::vector<Frame, GcAllocator<Frame>>`, scanned via the stack-resident header), which is why it scores 3 rather than 1 despite the small file count — the band description, not the file range, governs.
  - *Heat 4* — `Interpreter.cpp`/`.h` are the two hottest files in the tree; the tail-call and mark work of the last month lives exactly here.
- **Problem** — `Frame` (`Interpreter.h:100-158`) is a **shallow** module in the precise sense: its interface *is* its implementation. It is a tagged union of everything — a 13-value `Kind` enum and ~20 fields (`Exprs, Idx, Done, Saved, Begin0, ThenE, ElseE, Let, Def, DefEnv, RecScope, AppLoc, SetId, WcmValE, WcmResultE, WcmKeyV, Callee, Marks, Env`), of which each concrete frame uses a handful. Nothing is hidden; every reader must know which fields are live for which `Kind`. The behaviour that gives those fields meaning lives elsewhere, in the 275-line `continueStep` switch (`Interpreter.cpp:143-418`), so understanding one continuation kind means bouncing between the struct, the push-site, and the switch arm. That is the "understanding one concept requires bouncing between modules" smell and the "interface as complex as the implementation" smell at once.
- **Deletion test** — Delete the fat `Frame` + the switch and give each kind its own small type exposing a `resume(Interpreter&)` transition: complexity **concentrates**. Each arm's ~10-30 lines move onto the type that owns exactly the fields it reads; `continueStep` collapses to a one-line dispatch. It does not merely move — the shared-struct-plus-switch coupling is *dissolved*, not relocated. Strong signal.
- **Solution** — Introduce a continuation type per `Kind`, each carrying only its own fields and a `resume(Interpreter&)` (the body of today's switch arm). `continueStep` becomes dispatch over the top continuation. Keep a common header — at minimum `Marks` (read by `snapshotMarks` for *every* frame, `Interpreter.cpp:602`; written by `setMark` on the top activation, `:389,392`) and the `Callee`/reuse concept for the three tail-reusable activation kinds (`Call`, `WcmMark`, `Halt` — the predicate at `:387,559`). The exact representation (a `std::variant` payload vs a GC-allocated polymorphic base vs a transition table) is the step-4 design question; every candidate design must keep `Kont` GC-scannable.
- **Benefits** — *Leverage*: the machine driver (`run`, `continueStep`) shrinks to dispatch; each kind is independently readable and independently testable. *Locality*: a new continuation kind is one new type, not edits in three coupled places. *Test surface*: today the only way to exercise `LetBind`'s resume is to parse and run a `let-values` program end to end; a per-kind `resume()` can be unit-tested against a constructed continuation, and the bounded-space invariant (`getPeakKont`) is pinned before anything moves.

```mermaid
graph LR
  D[run/continueStep driver] --> SW{switch on Top.K, 13 arms}
  SW -.-> F[fat Frame: 20 fields, 13 kinds]
  P1[visit push-sites x14] -.-> F
  SW -.-> AP[applyProcedure tail-reuse reads Top.K/Callee/Marks]
```

Above: one driver switches on a tag into a fat shared struct that every push-site and the tail-call logic also reach into (dashed = all internal, nothing hidden).

```mermaid
graph LR
  D[run/continueStep driver] --> R[Top.resume]
  R -.-> K1[SeqK]
  R -.-> K2[AppK]
  R -.-> K3[LetBindK]
  R -.-> K4[WcmValK ...]
  H[common header: Marks + activation reuse] --> R
```

Below: the driver calls one `resume()` interface; each continuation kind hides its own fields and transition; only the universal marks/activation-reuse header is shared.

---

### formal-deep-interface — give the `Formal` tag real behaviour · Strong · score 21/25

- **Files** — `src/include/AST.h:489` (the 3-class `Formal` hierarchy whose whole virtual interface is `clone()`+`getType()`), re-dispatched at `Interpreter.cpp:85` (`formalsAccept`), `:466` (duplicated closure-arity check), `:512` (arg-binding), `AST.cpp:237` (`Lambda::dump`), and `AnalysisFreeVars.cpp:43`. Estimate: ~3 files.
- **Score — 21/25** — *Leverage 4*: five sites decode one tag; a deep `accepts`/`bind`/`collectNames` collapses them and deletes the `applyProcedure`↔`formalsAccept` duplication. *Locality 4*: adding a formal kind (keyword args) becomes one subclass, not five edits. *Blast radius 1*: 3 internal files, no published interface. *Heat 4*.
- **Problem** — `Formal` carries zero behaviour; every operation is a caller switching on `getType()` and `static_cast`-ing down. It is the most shallow module in the tree by the interface-vs-implementation measure. A live bug rides along: `Interpreter.cpp:514,521,534` bind with `auto LF = static_cast<const ast::ListFormal &>(F)` — `auto` deduces a **value**, deep-copying the formal's `SmallVector<Identifier>` on every application; `formalsAccept` (`:85`) uses `auto&` and proves the fix.
- **Deletion test** — Replace the tag + `getType()` with virtuals: five scattered switches concentrate into three method bodies next to the data. Concentrates.
- **Solution** — `bool accepts(size_t) const`, `void bind(...) const`, `void collectNames(set&) const`, `void dump(...) const` on `Formal`, overridden per subclass.
- **Benefits** — *Leverage*: arity/binding/dump stop re-dispatching; the closure path reuses `accepts`. *Locality/test surface*: each formal kind's behaviour is unit-testable directly on the node.
- **Recommendation strength** — Strong. Runner-up **candidate**; the natural next firing.

```mermaid
graph LR
  C1[formalsAccept] -.-> T[Formal::getType tag]
  C2[applyProcedure arity] -.-> T
  C3[arg-binding] -.-> T
  C4[Lambda::dump] -.-> T
  C5[AnalysisFreeVars] -.-> T
```

```mermaid
graph LR
  C1[callers] --> I[Formal.accepts/bind/collectNames/dump]
  I -.-> S1[ListFormal]
  I -.-> S2[ListRestFormal]
  I -.-> S3[IdentifierFormal]
```

---

### parse-form-combinators — collapse the form prologue behind combinators · Worth exploring · score 20/25

- **Files** — `src/Parse.cpp` (form parsers `parseLambda:707`, `parseCaseLambda:759`, `parseSetBang:981`, `parseIfCond:1105`, `parseLetValues:1238`, `parseWithContinuationMark:1164`, `parseVariableReference:1033`, `parseLinklet:628`, …), `src/include/Parse.h`. Estimate: ~2 files.
- **Score — 20/25** — *Leverage 4*, *Locality 4*, *Blast radius 2* (the parser + its unit test), *Heat 4*.
- **Problem** — Every keyword-led form opens with the identical `getPosition / gettok(LPAREN) / rewindTo / gettok(keyword) / rewindTo` prologue: the LPAREN guard appears 22×, `rewindTo(Start)` 53×, the `if (!hadError) parseError` emit-once idiom 18×. `parseExpr` (`Parse.cpp:211`) is a 14-alternative linear try-chain that re-lexes the leading `(`+keyword per alternative. The per-`parseX` interface hides almost nothing beyond that shared prologue.
- **Deletion test** — A `parseParenForm(keyword, body)` combinator plus a keyword→parser dispatch table concentrates the "(`(`, keyword) → parser" knowledge into one helper and one map. Concentrates — but there is a genuine ordering constraint (`Parse.cpp:281`: `quote` must precede application), so the table must model keyword-vs-application precedence; it is deep, not merely mechanical.
- **Solution** — `openForm`/`expect` combinators for the prologue; a keyword-keyed table for `parseExpr` dispatch.
- **Benefits** — *Leverage/locality*: the prologue lives once; adding a form is a table row. *Test surface*: `test_parse.cpp` already unit-tests parsers directly.
- **Recommendation strength** — Worth exploring — large internal churn in the hottest-but-most-fragile file.

```mermaid
graph LR
  E[parseExpr try-chain] -.-> P1[parseLambda + prologue]
  E -.-> P2[parseIf + prologue]
  E -.-> P3[parseLet + prologue]
  E -.-> P4[... 11 more, each re-lexing]
```

```mermaid
graph LR
  E[parseExpr] --> TB[keyword to parser table]
  TB -.-> B1[lambda body]
  TB -.-> B2[if body]
  TB -.-> B3[let body]
  OF[openForm/expect] --> B1
  OF --> B2
```

---

### bind-result-helper — one destructuring seam for let/letrec/define · Worth exploring · score 19/25

- **Files** — `src/Interpreter.cpp:54` (`bindValues`, used by `LetBind`/`LetRec`) and `:303` (the inline `Frame::Define` arm that re-implements it). Estimate: 1-3 files.
- **Score — 19/25** — *Leverage 3*, *Locality 4*, *Blast radius 1*, *Heat 4*.
- **Problem** — Multiple-values destructuring exists twice: `bindValues` and a hand-rolled copy in the `Define` arm, with cosmetically divergent diagnostics (`"let-values binding expected …"` vs `"define-values expected …"`) and a duplicated clone-loop (`:75` and `:323`). The concept "bind N results to N ids" has no single home.
- **Deletion test** — Fold `Define` onto a shared `bindResult(ids, value, contextLabel)`: concentrates; one arity/clone path, one place for the wording.
- **Solution** — A single helper parameterised by the context label so `let-values`/`define-values` diagnostics are preserved exactly.
- **Benefits** — *Locality/test surface*: one place to change and to FileCheck-pin the arity-mismatch messages.
- **Recommendation strength** — Worth exploring — small and contained; lower leverage than the pick.

```mermaid
graph LR
  L[LetBind/LetRec] --> BV[bindValues]
  DF[Define arm] -.-> IN[inline copy: divergent text]
```

```mermaid
graph LR
  L[LetBind/LetRec] --> BR[bindResult ids,value,label]
  DF[Define arm] --> BR
```

---

### environment-deepen — one scope module owning its own lifetime · Speculative · score 18/25

- **Files** — `src/include/Environment.h:8` (`Environment` leaks its `std::map` via `begin()/end()`), `Environment.h:41` (`Scope`, a method-less struct), `Environment.cpp:19` (chain ops as free functions), and `Interpreter.cpp:37,44,208` (the interpreter owns cycle-breaking via `AllScopes`). Estimate: ~4 files.
- **Score — 18/25** — *Leverage 4*, *Locality 4*, *Blast radius 2*, *Heat 2* (cold — YAGNI docks it).
- **Problem** — The "environment" concept is four fragments: a map-leaking wrapper, a bare `Scope`, free chain-functions, and — the tell — a lifetime/cycle-breaking policy that lives in the *client* (`~Interpreter` clears `AllScopes`), not in the module. The abstraction does not own its own invariant.
- **Deletion test** — Borderline. Folding chain-walking into `Scope` methods **and** moving scope allocation + teardown into an arena concentrates the policy; merging only the free functions would just move it. The win depends on the arena absorbing teardown.
- **Solution** — A scope/arena module with `contains()`, arena ownership, pointer-identity keys, and teardown responsibility; delete the leaked map surface.
- **Benefits** — *Locality*: lifetime policy stops straddling module and client. Couples to the in-progress `Value`/GC migration, so best scheduled alongside it.
- **Recommendation strength** — Speculative — cold code; coordinate with GC work.

```mermaid
graph LR
  I[Interpreter owns AllScopes teardown] -.-> SC[Scope struct]
  FF[envExtend/envLookup/envSet free fns] -.-> SC
  EN[Environment leaks its map] -.-> SC
```

```mermaid
graph LR
  I[Interpreter] --> AR[Scope arena: extend/lookup/set/contains + owns teardown]
  AR -.-> SC[Scope nodes]
```

---

### visitor-defaults-dead-code — default the visitor, delete the dead pass · Speculative · score 16/25

- **Files** — `src/include/ASTVisitor.h:11` (29 pure virtuals), `src/AnalysisFreeVars.{cpp,h}` (a 221-line `ASTVisitor` subclass instantiated nowhere — only referenced by `CMakeLists.txt:26`), `AST.h:649` (`Lambda::findFreeVariables` declared, never defined), and `Interpreter.cpp:776-842` (17 byte-identical `deliver(clone())` self-quoting overrides). Estimate: ~3-5 files.
- **Score — 16/25** — *Leverage 3*, *Locality 3*, *Blast radius 2*, *Heat 3*.
- **Problem** — Two smells bundled: dead weight (`AnalysisFreeVars` + the undefined declaration are compiled/declared but unreachable) and a too-wide interface (29 pure virtuals forcing an override in every visitor, one of which is dead; plus 17 identical value-node overrides that a single `visitSelfQuoting` hook would collapse).
- **Deletion test** — Removing the dead pass concentrates (pure removal). Collapsing the 17 overrides concentrates. Both good; but the dead-code half is cleanup, not deepening.
- **Solution** — Default the pure virtuals to no-ops; remove `AnalysisFreeVars` and the undefined declaration; optionally add a `visitSelfQuoting(ValueNode const&)` hook.
- **Benefits** — *Leverage*: adding a node kind stops forcing a no-op override in a dead visitor. *Test surface*: removal is verified by the existing suite staying green.
- **Recommendation strength** — Speculative — lowest leverage; the deepening and the cleanup are really two changes.

```mermaid
graph LR
  V[ASTVisitor: 29 pure virtuals] -.-> IN[Interpreter]
  V -.-> AF[AnalysisFreeVars: dead]
  I17[17 identical value overrides] -.-> IN
```

```mermaid
graph LR
  V[ASTVisitor: defaulted no-ops] --> IN[Interpreter]
  IN --> H[visitSelfQuoting hook]
  H -.-> M[17 value kinds]
```

---

## Dropped

| Candidate | Dropped because |
|---|---|
| `runtime-primitive-boilerplate` | Not a deepening — `Runtime::callFunction` (`Runtime.cpp:513`) is already a deep module: a one-line hash-map dispatch over `RUNTIME_FUNC`-registered polymorphic primitives. Leverage 1: the only smell (per-primitive `clone()`/`accept()` boilerplate, `FIXME` arithmetic error paths) would move complexity, not concentrate it. Recorded to prevent re-proposal. |
| `value-handle-deepen` | The `Value` handle (`Value.h:13`) is a deliberately thin `unique_ptr<ValueNode>` wrapper for the in-progress M2/GC value-model migration (documented at `Value.h:8`). Its shallowness is transitional by design; deepening now would contradict work in flight. Recorded to prevent re-proposal. |

## Too large to automate

None this run. No surviving candidate scored blast radius 5. The largest, `frame-per-kind-continuation`, is blast radius 3 — big in lines but contained to two private source files plus one test, with no published interface crossed — so it is one-PR work.

## Pick

**`frame-per-kind-continuation`, 22/25.** With `value-printing-raw-ostream-seam` reconciled to `landed` (PR #141 merged 2026-09-03), it is the top-scoring surviving `proposed` candidate. It clears every hard filter: leverage 5 (not 1), blast radius 3 (not 5), contradicts no ADR (there are none), status `proposed` (not landed/rejected/dropped/in-flight), and its behaviour is pinnable — `test/unit/test_interpreter.cpp:40-81` already asserts the bounded-`getPeakKont` tail-call invariant a continuation refactor could break, and `with-continuation-mark` semantics are FileCheck-pinned in integration `.rkt`s.

It outranks the runner-up **candidate**, `formal-deep-interface` (21/25), by a single point — **the top two are within 1 point**, so this pick was close and `formal-deep-interface` is the natural next firing. `frame-per-kind-continuation` wins on leverage (5 vs 4, doubled in the total): its deepening pays back across the entire 13-kind CEK loop and unblocks the delimited-continuation follow-ups (issue #11), whereas the runner-up, though lower-risk, pays back only across the five `Formal`-tag decode sites.

The prior firing's note that this candidate "wants characterization tests first" is satisfied by the method itself: step 5 pins the bounded-space and mark invariants with tests, watches them pass on the unrefactored machine, and only then moves the frames.

## Design

Produced at step 4 by three parallel design-it-twice sub-agents, each briefed for a radically different interface and all bound by the hard GC constraint (the `Kont` buffer must stay Boehm-scannable). A fourth sub-agent, which authored none of them, adjudicated against the fixed criteria (depth → locality → seam placement → test surface → blast radius). The three proposals and the verdict follow.

The three cross-kind seams every design had to preserve: (1) `Marks` — read by `snapshotMarks` on *every* frame, written by `setMark` on the top activation; (2) `Callee` — set on whichever of Call/WcmMark/Halt is the reused tail activation; (3) the reusable-activation predicate (`top is Call|WcmMark|Halt`) and the `Halt` terminator.

## Design A — minimal-surface variant

`Frame` becomes a universal header (`ast::MarkFrame Marks; std::unique_ptr<ast::ValueNode> Callee;`) plus a `std::variant<Seq, IfBranch, App, MkValues, LetBind, LetRec, Define, Set, WcmKey, WcmVal, Halt, WcmMark, Call>` payload — each alternative a small struct owning only its own fields. `continueStep` collapses to `std::visit([&](auto &k){ step(k, std::move(v)); }, Kont.back().K)`. The three tail-reusable activation kinds are ordered last so `isReusable()`/`isHalt()` are single unsigned index compares (`index() >= variant_size - 3`), pinned by `static_assert`s. `Kont` stays `std::vector<Frame, GcAllocator<Frame>>` unchanged: the variant is stored inline, so every owned value stays inside the scanned buffer — no heap indirection, ~45% smaller frame (variant sized to the largest arm, not the sum of all fields). Adding a kind: one struct + one variant arm + one `step` overload; the `size-3` derivation keeps the predicates correct with no constant edits. Trade-off: `std::visit` lowers to an indirect call through a dispatch table rather than a dense jump table (escape hatch: `switch (K.index())` + `std::get<N>`); the copy-out-before-`Kont`-mutation contract is unchanged from today but not compiler-enforced. Migration is self-checking — every stale `.K == Frame::Halt` fails to compile because the kinds are now types.

## Design B — polymorphic GC-node base

A new `Continuation` base class with a virtual `resume(Interpreter&)`, one subclass per kind, and an intermediate `ActivationCont` base owning `Callee` for the three reusable kinds; `Kont` becomes `std::vector<Continuation*, GcAllocator<Continuation*>>`, each cell `GC_MALLOC`'d via a `pushK<T>()` adapter. Dispatch is `Kont.back()->resume(*this)`; the seam predicates use the project's own `isa<>/classof` RTTI (`isa<ActivationCont>`, `isa<HaltCont>`) — an integer range-compare on a `ContKind` tag, zero vtable loads. Its distinctive leverage is future first-class continuations (issue #11): capturing a continuation is a pointer-slice copy of the stack rather than a deep clone, and a delimited prompt is one additive subclass. GC strategy: pointees are `GC_MALLOC`'d (scanned arena), the pointer vector is a scanned root; `popK`/`abortEval` must null the dead slots first to avoid false retention from the pointer buffer's stale tail. Trade-offs, and they are significant: per-step pointer-chase + vtable indirection and worse cache locality on the hot path; a `GC_MALLOC` on every non-tail push (allocation churn on deep non-tail recursion); the transition overrides need `friend`/passkey access back into the interpreter; and — decisively for sequencing — because popped cells are reclaimed without running `~Continuation`, B presupposes the POD-of-words value model that the M2/GC migration is *still landing* (`Value.h` says values are still `unique_ptr`-owned this phase), or else a fragile explicit-destructor bridge that becomes wrong the moment a frame can be captured. B is "best co-sequenced after the value-model migration," i.e. not now.

## Design C — co-located transition on value frames

`Frame` keeps its single concrete value type in `std::vector<Frame, GcAllocator<Frame>>` (GC-safe by construction, unchanged storage) and is split into a universal header (`Kind K; ast::MarkFrame Marks; EnvPtr Env; std::unique_ptr<ast::ValueNode> Callee; bool isReusableActivation()`) plus per-kind payload structs *nested inside `Interpreter`*, each owning its fields and two static-member transitions (`push`, `resume`). `continueStep` is a `default`-less 13-line switch delegating one line per kind (`case Kind::App: return App::resume(*this, Top);`) — a dense jump table with inlinable arms (cheapest hot path), and the missing `default` under warnings-as-errors makes a new unhandled kind a compile error. Nesting the structs in `Interpreter` both satisfies the clang-tidy static/anon-namespace constraint (they are member functions, not file-scope free functions) and grants private access without `friend`s. Honest near-term shape: the payloads land as **grouped members** (`Seq seq; App app; …` all present) — a pure, behavior-identical field-regrouping diffable arm-by-arm against today's switch, with the anonymous-**union** compaction (which brings the size win *and* type-enforced field validity) deferred to a separate slice because it requires four special-member functions (ctor/dtor/move-ctor/move-assign, the last instantiated by `abortEval`'s `erase`) that buy nothing on the hot path. Trade-off: until the union lands, `Frame` is not smaller (it is the sum of payloads) and field validity is not type-enforced — the deepening half-lands, with the type-safety and size dividend postponed.

### Adjudication

**Winner: Design A (minimal-surface variant).** It is the only design that lands the *whole* dividend in this one PR — the driver collapses to a single `std::visit` dispatch **and** each kind's fields become type-enforced (a stale `Seq` frame cannot touch `WcmKeyV`) **and** the frame shrinks (~45%, sized to the largest arm) — all on the *unchanged* `std::vector<Frame, GcAllocator<Frame>>` inline storage, with `~Frame` running normally and no dependence on the still-landing M2/GC value model. The criteria separate the designs cleanly at criterion 1; this is a decision, not a bail-out.

**Per-criterion (A vs B vs C):**

1. **Depth (delivered *this* PR).** **A** — driver becomes one `std::visit`; per-kind info-hiding + size + type-safety all land now. **B** — driver also collapses to one virtual `resume()`, but its *distinguishing* depth (first-class continuations, issue #11) is deferred — the design's own words are "not now." **C** — `continueStep` stays a **13-arm switch** (the driver still learns all 13 units, as today) and the anonymous **union** that carries type-safety + size is pushed to "a separate slice"; C itself says the deepening "half-lands." A > B > C.
2. **Locality.** A: a kind's data + transition sit on one variant arm + `step` overload — one-type edits. B leaks a false-retention null-out into `popK`/`abortEval` (driver code, not the kind). C co-locates transitions but, without the union, not field *validity*. A best.
3. **Seam placement.** B is cleanest here — `ActivationCont` groups exactly the three reusable kinds (`isa<ActivationCont>`/`isa<HaltCont>`). A and C carry `Callee` in the universal header (dead on 10 of 13 kinds) and gate reuse by an index-range/`isReusableActivation()` check. Both A's and C's seams are real (two live activations: the Marks header on every frame, the reuse predicate at `applyProcedure`/`WcmVal`), not hypothetical. Minor edge to B — but subordinate to 1–2, where A already leads.
4. **Test surface.** All three preserve the existing pins (`getPeakKont`, wcm bounded-space, the 8 `with-continuation-mark*.rkt`). B's `resume()` overrides need `friend`/passkey plus a `GC_MALLOC`'d cell to exercise in isolation; A/C dispatch functions have private access and are the easier to reach. Slight edge A/C.
5. **Blast radius.** A: 3 files (`Interpreter.h`, `Interpreter.cpp`, `test_interpreter.cpp`), no published interface, no `main.cpp`, no integration-output change, no allocation-model change — well under the >6 bail. C: smallest/safest of all (pure behaviour-identical field-regrouping) — but only because it defers the risky half. **B is disqualifying here**: it changes the allocation model (pointer vector + `GC_MALLOC` per non-tail push), adds a false-retention hazard (must null dead slots), and — decisively — reclaims popped cells **without running `~Continuation`**, which presupposes the POD-of-words value model that M2 is *still landing* (`Value.h`: values are `unique_ptr`-owned this phase). In an unattended PR mid-migration that either leaks the owned `ValueNode`s or needs a fragile throwaway bridge.

**Runner-up: Design B.** B is A's nearest rival on the *deciding* axis — it, too, collapses the driver to a single dispatch and gives each kind a real type owning only its own fields, with the cleanest seam (`ActivationCont`) and the strongest future leverage (capturing a continuation becomes a pointer-slice, not a deep clone). It lost because (i) at criterion 1 that distinguishing leverage is **deferred** (issue #11; "best co-sequenced after the value-model migration, i.e. not now"), so it delivers no more *this-PR* depth than A while A additionally banks the size + type-safety dividend; and (ii) at the criterion-5 tiebreak B is the *only* design that fails all three GC hazards (allocation-model change, false retention, dtor-skipping) against an unfinished M2 — unfit for a single unattended PR. (The framework files those three hazards under criterion 5, so they do not inflate B's criterion-1 score — B genuinely ranks second on depth, then loses on blast radius.) **C places third**: it fails at criterion 1 itself (keeps the 13-arm switch, defers the union), trading depth for the safest possible diff.

**Winner's test-first hazard list (pin FIRST, watch green on the *unrefactored* machine, then move the frames):**

1. **Reuse/Halt predicate ↔ variant ordering.** A replaces `K == Call || K == WcmMark || K == Halt` (`Interpreter.cpp:558`, `:386`) and the Halt terminator (`:146`) with unsigned index compares assuming the three activation kinds are the last three arms. Wrong ordering silently breaks tail-call reuse or mis-reuses a non-activation frame. *Pin:* `static_assert`s in `Interpreter.h` that `Call`/`WcmMark`/`Halt` occupy the last three variant indices (compile-time; no runtime probe — `Frame` is `Interpreter`-private). *Already covering end-to-end:* `test_interpreter.cpp:40-50`, `:69-77` (`Deep == Shallow`, `Deep < 16`).
2. **Copy-out-before-Kont-mutation (A's one un-enforced contract — inherited, not new).** `std::visit` binds `k` as a reference *into* `Kont.back()`; today's arms already hold `Frame &Top` the same way. Any arm that calls `pop_back` / `emplace_back` / `erase` / `evalBody` / `applyProcedure` / `abortEval` must extract every field it needs from `k` first — `emplace_back` can reallocate the `GcAllocator` buffer and invalidate `k` even without a pop. Sharp sites: `WcmKey` (pop → emplace `WcmVal`, `:354-359`), `WcmVal` (pop → maybe emplace `WcmMark`, `:386-393`), `App` (`applyProcedure` may emplace `Call`, `:208`), and `LetRec` which calls `abortEval` while its frame is *still live/unpopped* (`:275-279`). "Dispatch by value" is **not** a uniform fix — `Seq` (non-last), `LetRec`, and the `App`/`MkValues`/`LetBind` accumulation arms legitimately mutate `k` in place and keep the frame. *Pin:* run this PR's gate under **both** the `asan` and `ubsan` presets (they exist), and adopt an arm-by-arm review rule: no read of `k` after any Kont mutation. Exhaustiveness comes free — a missing `step` overload is a `std::visit` compile error (A keeps the same guarantee C claims from its default-less switch).
3. **wcm tail-position mark sharing.** `WcmVal` installs the mark onto the top frame when it is Call/WcmMark/Halt (`setMark`, `:389`) else pushes a fresh `WcmMark` — the *same* predicate as `applyProcedure`, so A must derive both from one index check. *Already pinned:* `test_interpreter.cpp:303-313` (`wcmTailLoopPeak`: bounded space) and `:315+` (same-key mark *replaces*, not accumulates). *Add if absent:* confirm one `with-continuation-mark*.rkt` asserts `current-continuation-marks` returns marks from **multiple** frames, exercising `snapshotMarks` reading the universal-header `Marks` on every frame (`:598-605`).
4. **let/letrec/define multiple-values arity + abort path.** `LetBind`/`LetRec`/`Define` (`:236`, `:269`, `:297`) destructure N results into N ids and `abortEval` on mismatch; `Define`'s 2-into-2 clone-loop is at `:323-329`. The abort erases Kont while the frame is live (hazard 2). *Gap found:* the arity-mismatch **diagnostics are not currently FileCheck-pinned** (`let-values4.rkt`/`define-values2.rkt` assert success values only). *Add:* three `.rkt`s pinning the exact messages — `"let-values binding expected …"`, `"define-values expected N values, got M"`, and the letrec-values mismatch — so the divergent wording survives the move.
5. **begin0's saved value.** `Seq` with `Begin0` saves the first expr in `Saved` and *persists* the frame to the end (`:151-153`, `:173-180`); the tail-position frame-drop is guarded `IsLast && !Top.Begin0` (`:156`) and must stay intact in A's `Seq` arm. *Covering:* `begin0.rkt`. *Confirm/extend:* it exercises ≥2 trailing effectful expressions after the saved value. (No tail-position begin0 test needed — begin0's last expression is not in tail position by construction.)
6. **`Callee` stays in the universal header.** Keep A's placement: the reuse path writes `Enc.Callee = std::move(Op)` onto a Halt/WcmMark/Call frame (`:562`), and `Callee` is also read at push (`:572`). Moving it into the three activation arms is tempting (deeper) but would force a `std::visit`/`get` at `:562`, `:572` and the two `setMark` sites — out of scope for this PR.


