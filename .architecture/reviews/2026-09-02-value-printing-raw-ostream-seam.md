# Architecture review — nora — 2026-09-02

**Branch**: `sym/nora/routine/refactor-audit/01M1GPA0JP` — **adopted** (all four conditions held at preflight: non-default, 0 commits ahead of `origin/main`, no upstream, unpublished on origin); not renamed, so the PR head is this name rather than `pm-deepen/<slug>`.
**Scope**: The interpreter core under `src/` (the pipeline SourceStream → Lex → Parse → AST → Interpreter → Runtime), weighted toward the git hot spots `src/Parse.cpp`, `src/include/AST.h`, `src/Interpreter.cpp`, `src/AST.cpp`, `src/Lex.cpp` (each ~13–17 commits, last touched 2026-07-02). The MLIR scaffold under `src/mlir/` and `src/include/nir/` is out of scope: it is opt-in, unused by the interpreter, and the project guide forbids adding it to the default build path.
**Picked**: `value-printing-raw-ostream-seam` — see [PR #141](https://github.com/pmatos/nora/pull/141) and `.architecture/backlog.md`.
**Degradations**: none. `codebase-design` loaded; a read-only `Explore` sub-agent performed the reconnaissance; `gh` is authenticated; the quality gate (`ctest --preset release`, 21 tests wrapping 88 `.rkt` integration tests) is green at baseline.

**Diagram convention**: solid edges are the module's **interface** (what a caller must cross); dashed edges are **implementation** internal to the module.

---

## Candidates

### `value-printing-raw-ostream-seam` — Route value printing through one injected `raw_ostream` seam · Strong · score 22/25

- **Files** — `src/include/AST.h:118` (the `virtual void write() const = 0` on `ValueNode`) and its ~16 overrides across `src/include/AST.h`, `src/AST.cpp` (`Symbol:85`, `Keyword:95`, `Char:104`, `String:113`, `Integer:192`, `Lambda:250`, `CaseLambda:274`, `Values:453`, `VariableReference:481`, `Void:544`, `List:581`, `Vector:622`, `QuotedExpr:658`), `src/include/ASTRuntime.h` + `src/ASTRuntime.cpp` (`Closure:29`, `CaseLambdaClosure:45`, `ContinuationMarkSet:112`), and the single external caller `src/main.cpp:80`. File-count estimate: **~5–6** (the four production files plus `test/unit/test_parse.cpp` for the pinning test).
- **Score** — **22/25**
  - Leverage **5**: the deepening pays back across all ~16 `write` sites at once (each stops choosing its own sink) *and* removes a whole class of test setup — value-printing behaviour that is integration-only today (subprocess + FileCheck) becomes assertable in-process via `llvm::raw_string_ostream`.
  - Locality **4**: the "which values are self-quoting" rule (today duplicated in `QuotedExpr::write` at `AST.cpp:662-663` and restated in ~6 header comments) and the choice of output channel both concentrate into one place.
  - Blast radius **2**: 5–6 files, a module and its direct caller; `write()` is internal to the `norac` binary, not a published/exported interface — no wire format or CLI contract changes.
  - Heat **4**: `AST.h`/`AST.cpp` are among the hottest files in the tree.
- **Problem** — Value printing is not a module at all; it is scattered across ~16 node overrides that each hard-wire their own output channel. There are **three** unsynchronised sinks: `std::cout` (most nodes), `llvm::outs()` (`RuntimeFunction::write`, `AST.h:980-982`), and C stdio via `gmp_printf` (`Integer::write`, `AST.cpp:192`). The header already documents the hazard at `AST.h:393-395` ("Use `std::cout` … `llvm::outs()` buffers independently and would reorder") and fixed `BooleanLiteral`, but the other two channels remain — so a result mixing a runtime function or integer with `std::cout` output can interleave out of order. The interface (`write()` to an ambient global) is *shallower than shallow*: it hides no dependency and its side effect cannot be observed except through a subprocess.
- **Deletion test** — Delete the ambient-`std::cout` style and force every writer through one passed `raw_ostream&`: complexity **concentrates** into one seam and one self-quoting predicate, and the three-channel interleaving hazard vanishes. Concentrates, not moves — a genuine deepening candidate.
- **Solution** — Change the interface to `virtual void write(llvm::raw_ostream &OS) const`, thread `OS` through all ~16 overrides, replace `gmp_printf` with `OS << asString()` (the method already exists on `Integer`), retire `llvm::outs()` in `RuntimeFunction::write`, and have `main.cpp` pass a single `llvm::outs()` for both its own result line and the trailing newline so nothing interleaves. Extract one `bool ValueNode::isSelfQuoting() const` to replace the duplicated `isa<>` chain. Behaviour-preserving: `Closure::write`/`CaseLambdaClosure::write` stay empty (their empty output is load-bearing and pinned by tests); the printed forms are unchanged.
- **Benefits** — **Leverage**: one interface change pays back across 16 sites and unlocks in-process assertions for the 20+ `quote*.rkt`/`char-*.rkt`/`values*.rkt` printed-form tests. **Locality**: self-quoting knowledge and channel choice live in one place; fix once, fixed everywhere. **Test surface**: this is the point — printed form becomes testable *through the interface* (`raw_string_ostream`) instead of past it (subprocess stdout), which is the `codebase-design` principle "accept dependencies, don't create them" applied to output.

```mermaid
graph LR
  M[main.cpp] --> W1[Symbol::write]
  M --> W2[Integer::write]
  M --> W3[QuotedExpr::write]
  W1 -.-> S1[std::cout]
  W2 -.-> S2[gmp_printf / stdio]
  W3 -.-> S3[llvm::outs / dup isSelfQuoting]
```

```mermaid
graph LR
  M[main.cpp] -->|OS| W[ValueNode::write OS]
  T[unit test] -->|raw_string_ostream| W
  W -.-> P[isSelfQuoting]
  W -.-> OS[single raw_ostream]
```

### `frame-per-kind-continuation` — Split the fat `Frame` struct into per-kind continuation types · Worth exploring · score 22/25

- **Files** — `src/include/Interpreter.h:84-146` (`struct Frame`, ~15 kind-specific fields valid only per `Kind`), `continueStep()`'s 13-arm switch `src/Interpreter.cpp:140-384`, and every `visit()` that constructs a frame (`Application:575`, `IfCond:608`, `LetValues:636`, `WithContinuationMark:699`, `SetBang`, …). File-count estimate: **2**, but the edit surface is the entire ~500-line evaluation core.
- **Score** — **22/25**. Leverage **5**: the CEK machine is the highest-behaviour interface in the tree. Locality **5**: each frame kind's push/resume logic would concentrate in one type instead of being spread across a fat struct, a switch arm, and a `visit`. Blast radius **3** (inverted → contributes 3): only 2 files, but "how much code the refactor touches" (the rubric's definition, with the file table demoted to a sanity check) is the whole evaluation loop — the largest code volume of any candidate. Heat **4**.
- **Problem** — The field-validity invariant ("which of the ~15 fields are live for which `Kind`") lives only in the programmer's head and leaks to every construction and consumption site; understanding one form means reading its `visit`, its switch arm, and `applyProcedure`.
- **Deletion test** — Replacing the fat struct + switch with per-kind frame types (a `std::variant`, or a polymorphic `Frame::resume(Interpreter&)`) concentrates each kind's transition in one place. Concentrates.
- **Solution** — One small type per `Kind` holding only its fields, each with a `resume()` transition; `continueStep` becomes dispatch.
- **Benefits** — **Leverage** and **locality** are both maximal; the **test surface** improves because a frame's transition becomes unit-addressable. But this is the core machine and the riskiest change in the report — it wants characterization tests written first.

```mermaid
graph LR
  V[visit Application] --> F[Frame fat struct ~15 fields]
  V2[visit IfCond] --> F
  C[continueStep 13-arm switch] --> F
  F -.-> X[which fields valid?]
```

```mermaid
graph LR
  V[visit Application] --> FA[AppFrame]
  V2[visit IfCond] --> FI[IfFrame]
  C[continueStep dispatch] -->|resume| FA
  C -->|resume| FI
```

### `formal-deep-interface` — Give `Formal` a deep interface for arity, binding, and bound vars · Strong · score 21/25

- **Files** — `src/include/AST.h:480-565` (`Formal` exposes only `getType()` + `clone()`), with its `Type` enum re-dispatched via `switch`/`static_cast` at `src/Interpreter.cpp:85-95` (`formalsAccept`), `:430-450` (inline arity check, with *different* error text than `formalsAccept`), `:478-508` (argument binding), and `src/AST.cpp:213-244` (`Lambda::dump`). File-count estimate: **3** (`AST.h`, `AST.cpp`, `Interpreter.cpp`).
- **Score** — **21/25**. Leverage **4**: four dispatch sites collapse and a deeply-nested caller (`applyProcedure`) stops reaching past the seam. Locality **4**: adding a formal kind or changing arity/binding today forces edits in four places. Blast radius **1** (→5): 3 files, no published interface. Heat **4**.
- **Problem** — `Formal` is a shallow tag: its interface (`getType`) is as complex as knowing which subclass you hold, so every behaviour (arity acceptance, argument binding, bound-variable collection) is re-implemented at the call sites. A concrete latent inefficiency rides along: `auto LF = static_cast<const ast::ListFormal &>(F)` at `Interpreter.cpp:480/487/500` **copies** the formals `SmallVector` on every application (a `const&` won't compile because `operator[]` is non-const at `AST.h:539`).
- **Deletion test** — Replace the tag-plus-switch with virtuals on `Formal`: complexity concentrates into `Formal`. Concentrates.
- **Solution** — Add `bool accepts(size_t nargs) const`, an argument-binding method, and `SmallVector<Identifier> boundVars() const`; make `operator[]` const. Collapses `formalsAccept`, the inline arity check, the binding switch, and `Lambda::dump`'s switch, and lets a canonical arity-error message replace the two divergent ones.
- **Benefits** — **Leverage** across four sites, strong **locality**, and the copy bug disappears. **Test surface**: arity/binding become unit-addressable on `Formal`.

```mermaid
graph LR
  A[applyProcedure] --> T[Formal getType tag]
  FA[formalsAccept] --> T
  D[Lambda::dump] --> T
  T -.-> SW[switch + static_cast x4]
```

```mermaid
graph LR
  A[applyProcedure] -->|accepts / bind| F[Formal]
  D[Lambda::dump] -->|boundVars| F
  F -.-> I[IdentifierFormal]
  F -.-> L[ListFormal]
  F -.-> R[ListRestFormal]
```

### `parse-form-combinators` — Fold repeated parse prologue and diagnostic idiom into combinators · Worth exploring · score 20/25

- **Files** — `src/Parse.cpp`: the `getPosition → gettok LPAREN → rewindTo → gettok KEYWORD → rewindTo` prologue repeated in ≥10 functions (`parseDefineValues:498`, `parseValues:564`, `parseLambda:700`, `parseCaseLambda:752`, `parseBegin:874`, `parseSetBang:963`, `parseVariableReference:1017`, `parseIfCond:1087`, `parseWithContinuationMark:1148`, `parseLetValues:1220`), the `if (!hadError(S)) parseError(...)` idiom repeated ~15×, and the hand-rolled ordered dispatch in `parseExpr:211-292`. Plus `src/include/Parse.h`. File-count estimate: **2**, ~10–15 functions.
- **Score** — **20/25**. Leverage **4** (10+ functions simplify). Locality **4** (form-opening + error idiom concentrate). Blast radius **2** (→4): 2 files but many internal edits in the largest, most-churned file. Heat **4**.
- **Problem** — Every form parser threads `SourceStream` position/rewind manually, leaking the tokenizer's positional model to every caller. A latent correctness gap rides along: `VOID`/`RAISE_ARGUMENT_ERROR`/`PROCEDURE_ARITY_INCLUDES_C`/`MAKE_STRUCT_TYPE` are lexed as dedicated tokens but no expression parser consumes them, so `(void)` in expression position cannot parse.
- **Deletion test** — An `openForm(S, Keyword)` guard + `expect(S, TokType, msg)` combinator + a keyword→parser table concentrate the boilerplate. Concentrates.
- **Solution** — Behaviour-preserving extraction of the prologue and the `hadError`/`parseError` idiom into combinators; the `(void)` fix and error-regime convergence are documented follow-ups, not part of the extraction.
- **Benefits** — **Leverage** across ~10 parsers, **locality** for form-opening; **test surface** already strong (`test_parse.cpp` + dozens of `.rkt`).

```mermaid
graph LR
  P1[parseLambda] --> B[manual prologue + rewind]
  P2[parseBegin] --> B
  P3[parseLetValues] --> B
  B -.-> E[if !hadError parseError x15]
```

```mermaid
graph LR
  P1[parseLambda] -->|openForm/expect| G[form combinators]
  P2[parseBegin] -->|openForm/expect| G
  P3[parseLetValues] -->|openForm/expect| G
  G -.-> R[position + rewind + diagnostics]
```

### `bind-result-helper` — Unify multiple-values destructuring across let/letrec/define · Worth exploring · score 19/25

- **Files** — `src/Interpreter.cpp:54-82` (`bindValues`) vs the `Frame::Define` arm reimplementing the same "1 id → value / N ids → arity-checked `Values`" logic at `:278-313`, with different error text (`:69-72` vs `:297-301`). File-count estimate: **1–3**.
- **Score** — **19/25**. Leverage **3** (two call sites simplify). Locality **4** (a divergent-bug site is eliminated). Blast radius **1** (→5). Heat **4**.
- **Problem** — The multiple-values binding logic exists twice with divergent error text; the root cause is that `ast::Values` (`AST.h:828-865`) does double duty as surface syntax and runtime result, forcing both paths to `clone()` then `dyn_cast`. A shared gap: `(let-values (((x) (values 1 2))) x)` does not error because `size(ids)==1` binds whatever arrived.
- **Deletion test** — Extract `bindResult(Diag, Loc, Scope, Ids, value)` used by LetBind, LetRec, and Define: concentrates.
- **Solution** — Extract the helper; optionally introduce a distinct runtime `MultipleValues` value to remove the clone-then-downcast.
- **Benefits** — **Locality** (one binding rule), **leverage** across two sites; **test surface** via existing `let-values*/letrec-values*/define-values*` tests.

```mermaid
graph LR
  LV[bindValues let/letrec] -.-> R1[1-id/N-id + clone/dyncast]
  DF[Define arm] -.-> R2[1-id/N-id + clone/dyncast]
```

```mermaid
graph LR
  LV[bindValues] -->|bindResult| H[bindResult]
  DF[Define arm] -->|bindResult| H
  LR[letrec] -->|bindResult| H
  H -.-> R[one binding rule]
```

### `environment-deepen` — Fold Environment + Scope + cycle-breaking into one module · Speculative · score 18/25

- **Files** — `src/include/Environment.h` + `src/Environment.cpp` (`Environment` flat map, `Scope` chain, free functions `envLookup`/`envSet`/`envExtend`), and the Interpreter reaching past the abstraction to break cycles via `AllScopes` + destructor loop (`src/Interpreter.cpp:31-49`, `Interpreter.h:187-191`), plus `Identifier::operator<=>` (`src/AST.cpp:68-73`). File-count estimate: **4**.
- **Score** — **18/25**. Leverage **4**. Locality **4**. Blast radius **2** (→4): `operator<=>` affects every `std::map<Identifier,…>`. Heat **2**: `Environment.cpp` is cold (2 commits) — YAGNI docks this hard.
- **Problem** — Scoping is split across three layers; the Interpreter owns cycle-breaking that belongs to the scope abstraction. Two hidden inefficiencies: `envSet` calls `lookup` (which `clone()`s the value) purely to test membership, and `operator<=>` does a byte-wise compare even though `IdPool` guarantees pointer-unique backing strings. Dead surface: `envExtend`, `Environment::begin/end`.
- **Deletion test** — Fold the layers into one `Scope`/`Environment` with `contains()`, arena ownership, and pointer identity: concentrates.
- **Solution** — Add `contains()`; switch identity to interned-pointer comparison; move cycle-breaking into the scope arena; delete the dead surface.
- **Benefits** — **Locality** for scoping, **leverage** on set!/lookup; behaviour-neutral perf win.

```mermaid
graph LR
  I[Interpreter] --> E[Environment map]
  I --> S[Scope chain]
  I -.-> C[AllScopes + dtor loop]
  E -.-> CL[lookup clones on membership probe]
```

```mermaid
graph LR
  I[Interpreter] -->|lookup/set/contains| SC[Scope module]
  SC -.-> A[arena ownership + cycle breaking]
  SC -.-> PI[pointer-identity keys]
```

### `visitor-defaults-dead-code` — Default `ASTVisitor` no-ops and remove the dead free-vars pass · Speculative · score 16/25

- **Files** — `src/include/ASTVisitor.h:7-38` (27 pure virtuals, no defaults), the dead `src/AnalysisFreeVars.{cpp,h}` (never instantiated; `Lambda::findFreeVariables` at `AST.h:640` is declared but undefined), the ~15 empty self-evaluating overrides in `src/Interpreter.cpp:717-775`, and CMake entries. File-count estimate: **3–5**.
- **Score** — **16/25**. Leverage **3**. Locality **3**. Blast radius **2** (→4). Heat **3**.
- **Problem** — Adding an AST node forces edits to AST.h, ASTVisitor.h, Interpreter.h/.cpp, and the dead AnalysisFreeVars — 6+ files, one of them dead weight.
- **Deletion test** — (i) Deleting the dead pass makes complexity *vanish* (a cleanup, not a deepening); (ii) defaulting the visitor's virtuals to no-ops concentrates the "nothing to do" cases in the base — that half concentrates.
- **Solution** — Remove `AnalysisFreeVars.{h,cpp}` + `findFreeVariables` + their CMake entry; give `ASTVisitor` defaulted no-op bodies (or a `visitSelfEvaluating` hook).
- **Benefits** — **Leverage**/**locality** modest; mostly reduces per-visitor boilerplate and dead weight.

```mermaid
graph LR
  N[new AST node] --> A[AST.h]
  N --> V[ASTVisitor 27 pure virtuals]
  N --> D[dead AnalysisFreeVars]
```

```mermaid
graph LR
  N[new AST node] --> A[AST.h]
  N -->|override only what matters| V[ASTVisitor defaulted no-ops]
```

---

## Dropped

None. No candidate trips a hard filter: every candidate has leverage ≥ 3 (none fails the deletion test outright), none has blast radius 5, there are no ADRs to contradict (`docs/adr/` does not exist), the backlog is new, and every candidate's behaviour is pinnable by the existing Catch2 + lit suites.

| Candidate | Dropped because |
|---|---|
| _(none)_ | — |

## Too large to automate

None. The largest candidate (`frame-per-kind-continuation`) is blast radius 3, implementable in one PR (behind characterization tests); it is not blast radius 5.

## Pick

**`value-printing-raw-ostream-seam`, 22/25.** It ties `frame-per-kind-continuation` (also 22/25) at the top and is separated by the rubric's deterministic tie-break — **lower blast radius wins**: value-printing touches 5–6 files and no evaluation-core code (blast 2), whereas the Frame split rewrites the entire ~500-line CEK loop (blast 3). Three candidates sit within one point (22 / 22 / 21), so the pick was close:

- **Runner-up candidate: `frame-per-kind-continuation` (22/25)** — equal score, deeper structurally, but the riskiest change in the tree and correctly deprioritised by blast radius. It wants characterization tests first.
- **`formal-deep-interface` (21/25)** is the safest, most textbook "shallow tag → deep module" refactor and is the **natural next firing** after this one lands.

Value-printing was chosen on its own merits, not to avoid the others: it is the report's clearest *testability* win (it converts 20+ integration-only printed-form tests into in-process assertions — exactly the aim of the exercise), it fixes a real latent interleaving bug the codebase already documents, and it is well-pinned by the existing `quote*/char-*/values*` suites so a behaviour-preserving extraction is safe.

## Design

Produced by four parallel design sub-agents (radically different framings), then chosen by a fifth **adjudicator** that authored none of them, against the fixed criteria in strict priority order: **depth → locality → seam placement → test surface → blast radius**.

### The four proposals

- **A — Minimal surface.** `virtual void write(llvm::raw_ostream &OS) const = 0` on `ValueNode`, `OS` threaded through every override and every recursive child; a free `bool ast::isSelfQuoting(const ValueNode&)` defined once in `AST.cpp`; `Integer` uses `asString()`, `RuntimeFunction` drops `llvm::outs()`. No new files. ~6 files, ~40 mostly one-line sites. Correctly notes no `flush` is needed (verbose `std::cout` lines end in `std::endl` and precede the result).
- **B — Maximum flexibility (`Printer` context object).** New `src/include/Printer.h` with `struct Printer { llvm::raw_ostream &OS; }`; `virtual void print(Printer&)`; a **virtual** `isSelfQuoting()` hook overridden in the four self-quoting leaves; a documented growth path to add a `PrintStyle` enum later for `display`/`print`. ~7 files, ~50 sites.
- **C — Optimise for the common caller.** A's primitive **plus** a non-virtual, logic-free `std::string ValueNode::str()` convenience so tests/`main` can get a string without building a stream. ~6 files, ~40 sites.
- **D — Ports-and-adapters (rejects its own framing).** Argues that pulling printing out of the hierarchy into a free function/visitor is over-engineered — the printer-*policy* seam has only **one** adapter (the sole Racket print form), so it is hypothetical; a non-member dispatcher would hand-roll the vtable and lose locality, and the existing `ASTVisitor` spans 27 node kinds (~9 non-values = dead surface). Converges on A's member design, but pins the predicate **file-local** (`static` in `AST.cpp`) and deletes the two comments that become false once there is a single sink.

### Adjudication — winner: **D** (runner-up design: **A**)

D and A are the *same* `write(llvm::raw_ostream&)` member interface; the adjudicator separated them on the top-priority criterion. **Depth**: D keeps `isSelfQuoting` `static`/file-local, so the module's public surface is exactly one method — strictly less interface for the same behaviour than A's namespace-scope free function. **Locality**: A ≈ C ≈ D best (predicate in one place); B worst (the virtual hook disperses a closed, language-fixed set `{Integer, BooleanLiteral, Char, String}` across five classes for no gain). **Seam placement**: A ≈ C ≈ D place exactly the seam that varies — the *sink*, which has two real adapters today (`llvm::outs()` in `main`, `raw_string_ostream` in tests); B additionally scaffolds a one-adapter (hypothetical) mode seam. **Test surface**: all four unlock in-process assertions equally (`test_parse` links `AST.cpp` + `ASTRuntime.cpp`, not `Interpreter.cpp`, which is sufficient for value nodes). **Blast radius**: A smallest, D ≈ A, B largest.

**Runner-up design (A) lost** because its free-function predicate, as specified, risks publishing a `QuotedExpr`-internal policy at namespace scope (a depth criterion-1 loss), and it defers the mandatory stale-comment deletion, leaving the change internally inconsistent. Its one genuine edge — recognising no `flush` is needed — is a criterion-5 point, outranked by D's criterion-1 edge.

### The two recurring sub-decisions

- **`isSelfQuoting`: free function, file-local (`static` in `AST.cpp`).** The set is a closed, rarely-changing group of literals; one function gives single-site locality and adds nothing to the public interface. The virtual form (B) scatters it across five classes for extensibility nobody needs.
- **Neither `str()` nor `Printer` earns its keep today.** `Printer` services a mode seam with one adapter → hypothetical, and it is expensive/hard to reverse. `str()` is cheaper and reversible but is a *convenience, not a seam*; its sole present consumer (tests) is served by a two-line `raw_string_ostream`, and adding it creates a second way to do one thing (a criterion-1 cost) for marginal criterion-4 ergonomics. Add neither until a second consumer appears.

### Interface to implement (the adjudicated winner, with A's flush correction)

```cpp
// src/include/AST.h — ValueNode
virtual void write(llvm::raw_ostream &OS) const = 0;   // was: virtual void write() const = 0;
```

- Thread `OS` through all ~16 overrides (`AST.h` inline: `BooleanLiteral`, `RuntimeFunction`; `AST.cpp`: 13 bodies; `ASTRuntime.{h,cpp}`: `Closure`/`CaseLambdaClosure` stay empty, `ContinuationMarkSet`). Empty/no-op writers take an **unnamed** `llvm::raw_ostream &` to stay `-Wunused-parameter`-clean under `-Werror`.
- `Integer::write` → `OS << asString()` (drops `gmp_printf`); `RuntimeFunction::write` → `OS << "#<runtime:" << getName() << ">"` (drops `llvm::outs()`); `BooleanLiteral::write` → `OS << (Value ? "#t" : "#f")`.
- Compound nodes (`List`, `Vector`, `Values`, `QuotedExpr`) forward the same `OS` to children; `std::endl` → `'\n'`.
- The self-quoting rule (`isa<Integer, BooleanLiteral, Char, String>`) consolidated into its **only** caller, `QuotedExpr::write`. _Implementation note:_ the adjudicated design put this in a file-local `static isSelfQuoting()` helper, but the repo's `.clang-tidy` enables **both** `misc-use-anonymous-namespace` (rejects a `static` free function) **and** `llvm-prefer-static-over-anonymous-namespace` (rejects an anonymous-namespace one); the two checks conflict on any new file-scope free function, and the CI clang-tidy job is diff-scoped with `-warnings-as-errors`, so either form fails on a changed line. Inlining into the sole caller keeps a single home for the rule, adds nothing to `ValueNode`'s interface (the adjudicator's top criterion), and is exercised by the new test through `QuotedExpr`.
- `src/main.cpp:80-81` → `I.getResult()->write(llvm::outs()); llvm::outs() << '\n';`. No `flush` (the redundant `std::cout.flush()` from D is omitted per A's correct analysis).
- Delete the two comments made false by a single sink: `BooleanLiteral`'s `std::cout`-vs-`outs()` justification (`AST.h`) and `Symbol`'s interleaving note (`AST.cpp`).

**Test-first seam under test**: `ValueNode::write(raw_ostream&)`, asserted in `test/unit/test_parse.cpp` via `llvm::raw_string_ostream` on nodes built directly (e.g. `Integer`, `List`, `QuotedExpr`), pinning printed form and the self-quoting rule (`'7` → `7`, `'(1)` → `'(1)`).
