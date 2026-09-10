# Architecture deepening backlog

Persistent memory for the `pm-deepen` routine. One `## <slug>` entry per candidate ever seen; statuses change, rows stay. `### Run` blocks under an entry (or under `## Run log`) are history, not candidates.

## runtime-builtin-boilerplate

- **Status**: landed
- **Score**: 24/25 (leverage 5, locality 5, blast radius 2, heat 5)
- **Files**: ~3 estimated (`src/Runtime.cpp`, `src/include/Runtime.h`, `src/include/AST.h`); actual: 4 (those three plus `test/unit/test_interpreter.cpp`)
- **Modules**: `src/Runtime.cpp`, `src/include/Runtime.h`, `src/include/AST.h`
- **Summary**: Collapse the 18 hand-rolled `RuntimeFunction` subclasses — each re-rolling the same arity check, per-arg `dyn_cast` type prologue, and `clone()`/`accept()` tails — behind one deep `RuntimeFunction` seam that owns arity + argument-type dispatch, registering builtins as `{name, arity-spec, typed-handler}` data with handler cores unchanged. All builtins share one `AST_RuntimeFunction` node kind and one visitor overload, so the change never touches the enum, visitors, RTTI, or the caller's `nullptr`→diagnostic contract.
- **First seen**: 2026-09-10
- **PR**: #200 (branch `sym/nora/routine/refactor-audit/01M25W8YV7`, adopted) — **merged 2026-09-10**
- **Reason (picked)**: Top of the 2026-09-10 ranking at 24/25; outranks runner-up candidate `value-register-take-vs-borrow` (22/25) by 2 points. Fresh candidate from the M2/GC hot-spot scan; the "sibling functions repeat the same prologue" collapse.
- **Reconciled 2026-09-11**: `gh pr view 200` → MERGED (2026-09-10T20:00:54Z), `in-flight` → `landed`. No open architecture PR blocks the 2026-09-11 run.

### Run 2026-09-10 — complete

- **Outcome**: complete
- **Stopped at**: step 6 — PR #200 opened for `runtime-builtin-boilerplate`
- **Branch**: `sym/nora/routine/refactor-audit/01M25W8YV7` (adopted — non-default, no unique history, no upstream, unpublished on origin; kept the caller's name, not renamed)
- **Committed**: report `.architecture/reviews/2026-09-10-runtime-builtin-boilerplate.md`, reconciled backlog (`frame-per-kind-continuation` #191 MERGED → landed; three fresh candidates added), the registry refactor, and its test-first pins in `test/unit/test_interpreter.cpp`
- **Evidence**: `gh pr view 191/141` both MERGED; no open architecture PR blocked this run; quality gate green — release build warning-clean, `ctest --preset release` 55/55, `clang-format` clean, diff-scoped `clang-tidy` (clang DB, `-warnings-as-errors='*'`) clean
- **Degradations**: advisor rate-limited — step-4 adjudication self-done against the written designs; step-4 designs produced inline rather than by parallel sub-agents
- **Next**: review PR #200; the natural next firing is the runner-up candidate `value-register-take-vs-borrow` (22/25)

## value-register-take-vs-borrow

- **Status**: in-flight
- **Score**: 22/25 (leverage 4, locality 4, blast radius 1, heat 5)
- **Files**: ~1-2 estimated; actual: 3 (`src/include/Value.h`, `src/Interpreter.cpp`, `test/unit/test_environment.cpp`) — within the step-5 2× guard
- **Modules**: `src/include/Value.h`, `src/Interpreter.cpp`
- **PR**: #202 (branch `pm-deepen/value-register-take-vs-borrow`, created)
- **Summary**: The `Value` handle leaks its two representations through `get()`/`takeLegacy()`/`toShared()`; six `step` arms reflexively `takeLegacy()` — which clones a shared value — where a borrow or an into-`Value`-sink move would do, re-introducing the copies #195/#196 removed. Give `Value` a non-cloning borrow/into-sink path and route the read-only/pass-through arms through it.
- **First seen**: 2026-09-10
- **Reason (picked 2026-09-11)**: Top surviving `proposed` candidate at 22/25 once `runtime-builtin-boilerplate` (#200) landed — the prior run's recorded "natural next firing". Runner-up candidate this run is `formal-deep-interface` (21/25), within 1 point. Reconfirmed present: 8 `takeLegacy()` sites in `Interpreter.cpp` (6 step arms + 2 in `applyProcedure`), 5 of them avoidable clones (198/306/341/400/408); the 3 genuine `unique_ptr<ExprNode>` sinks (237/525/534) are the AST value-conflation, out of scope. #198 hardened the primitives without changing the call-site friction. Overlaps `bind-result-helper`.

### Run 2026-09-11 — complete

- **Outcome**: complete
- **Stopped at**: step 6 — PR #202 opened for `value-register-take-vs-borrow`
- **Branch**: `pm-deepen/value-register-take-vs-borrow` (created from `origin/main` and renamed to the slug — the firing branch `sym/nora/routine/refactor-audit/01M26RZFYA` was **not** adoptable: condition 4 refused it, `refs/remotes/origin/sym/nora/routine/refactor-audit/01M26RZFYA` exists, i.e. it is published on origin)
- **Committed**: report `.architecture/reviews/2026-09-11-value-register-take-vs-borrow.md` + design section, reconciled backlog (`runtime-builtin-boilerplate` #200 MERGED → landed; four fresh candidates added), the `Value::as<T>()` borrow seam and its test-first pins in `test/unit/test_environment.cpp`, and a seeded `CONTEXT.md`
- **Evidence**: `gh pr view 200` MERGED, no open architecture PR blocked this run; no `pm-deepen/*` slug branch existed on origin (collision check clear). Quality gate: GCC `release` (`-Werror -O3 -flto`) warning-clean + `ctest` 70/70; GCC `asan` clean + 70/70 + no ASan/LSan reports; Clang 22 `release` warning-clean; diff-scoped `clang-tidy` clean; `clang-format` 22.1.8 clean
- **Degradations**: advisor rate-limited only at the step-3 approach check (available at step-4 adjudication); the repo's auto-format hook targets `clang-format-22` (absent) — ran unversioned `clang-format` 22.1.8 manually, same version
- **Next**: review PR #202; the natural next firing is the runner-up candidate `formal-deep-interface` (21/25)

## continuation-mark-query-seam

- **Status**: proposed
- **Score**: 19/25 (leverage 3, locality 4, blast radius 2, heat 5)
- **Files**: ~3 estimated
- **Modules**: `src/Runtime.cpp`, `src/include/ASTRuntime.h`, `src/ASTRuntime.cpp`
- **Summary**: `ContinuationMarkSet` exposes only `getFrames()`; `continuation-mark-set-first` and `continuation-mark-set->list` both reach past the seam and re-implement the same innermost-first key scan. Offer `firstForKey(key)`/`allForKey(key)`. A bundled eq-drift (mark keys compared by symbol name via `valueEq`, while `eq?` compares identity) is scope creep, excluded from the estimate and left for a human.
- **First seen**: 2026-09-10
- **Reason (deprioritised)**: 19/25; smaller leverage than the pick, and the correctness-flavoured eq-drift should not ride along in a pure seam extraction.

## value-printing-raw-ostream-seam

- **Status**: landed
- **Score**: 22/25 (leverage 5, locality 4, blast radius 2, heat 4)
- **Files**: ~5-6 estimated (actual: 6 — the five modules plus `test/unit/test_parse.cpp`)
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/include/ASTRuntime.h`, `src/ASTRuntime.cpp`, `src/main.cpp`
- **Summary**: Route every `ValueNode::write` through one injected `llvm::raw_ostream` seam and a single self-quoting `isa<>` set, retiring the three unsynchronised output channels (`std::cout`, `llvm::outs`, `gmp_printf`).
- **First seen**: 2026-09-02
- **PR**: #141 (branch `sym/nora/routine/refactor-audit/01M1GPA0JP`, adopted) — **merged 2026-09-03**

## frame-per-kind-continuation

- **Status**: landed
- **Score**: 22/25 (leverage 5, locality 5, blast radius 3, heat 4)
- **Files**: ~3 estimated (actual: 5 — the two source files `src/include/Interpreter.h`, `src/Interpreter.cpp` plus three new `.rkt` arity pins; `test/unit/test_interpreter.cpp` was left untouched, its existing `getPeakKont`/wcm invariants already covered the refactor)
- **Modules**: `src/include/Interpreter.h`, `src/Interpreter.cpp`
- **Summary**: Split the fat multi-purpose `Frame` struct (13-value `Kind` enum, ~20 kind-specific fields) into one small per-`Kind` continuation type with a `resume()` transition, replacing the 13-arm `continueStep` switch with dispatch, while preserving the GC-scanned `Kont` buffer, the universal `Marks` header, and the Call/WcmMark/Halt tail-call reuse seam.
- **First seen**: 2026-09-02
- **PR**: #191 (branch `sym/nora/routine/refactor-audit/01M1MR2N1J`, adopted) — **merged 2026-09-04**
- **Reason (picked)**: Top surviving `proposed` candidate at 22/25 once `value-printing-raw-ostream-seam` landed. Within 1 point of the runner-up candidate `formal-deep-interface` (21/25).
- **Reconciled 2026-09-10**: `gh pr view 191` → MERGED, `in-flight` → `landed`.

### Run 2026-09-04 — complete

- **Outcome**: complete
- **Stopped at**: step 6 — PR #191 opened for `frame-per-kind-continuation`
- **Branch**: `sym/nora/routine/refactor-audit/01M1MR2N1J` (adopted — non-default, no unique history, no upstream, unpublished on origin; kept the caller's name, not renamed)
- **Committed**: report `.architecture/reviews/2026-09-04-frame-per-kind-continuation.md`, reconciled backlog, three `.rkt` arity pins, and the per-kind-continuation refactor
- **Evidence**: `value-printing-raw-ostream-seam` PR #141 reconciled MERGED → landed; no open architecture PR blocked this run; toolchain (LLVM 22 / GMP / libgc / Catch2) verified by `cmake --preset release`; quality gate green on release + asan + ubsan (40/40) and warning-clean on GCC + Clang; `clang-format` and diff-scoped `clang-tidy` clean
- **Next**: review PR #191; the natural next firing is `formal-deep-interface` (21/25)

## formal-deep-interface

- **Status**: proposed
- **Score**: 21/25 (leverage 4, locality 4, blast radius 1, heat 4)
- **Files**: ~3 estimated
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/Interpreter.cpp`
- **Summary**: Give the shallow `Formal` tag a deep interface (`accepts`, argument-binding, `boundVars`) so arity, binding, and dump stop re-dispatching on `Formal::Type` at five sites (grew from four: `formalsAccept`, the duplicated closure-arity check in `applyProcedure`, arg-binding, `Lambda::dump`, `AnalysisFreeVars`); also fixes the per-application `auto` value-copy of the formal's identifier vector.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 21/25, within one point of the pick; the safest textbook shallow→deep refactor and the natural next firing. Runner-up candidate this run.

## parse-form-combinators

- **Status**: proposed
- **Score**: 20/25 (leverage 4, locality 4, blast radius 2, heat 4)
- **Files**: ~2 estimated (~10-15 functions in the largest file)
- **Modules**: `src/Parse.cpp`, `src/include/Parse.h`
- **Summary**: Fold the repeated `getPosition/gettok/rewind` form prologue (22× LPAREN guard, 53× rewindTo, 18× emit-once idiom) and the `if (!hadError) parseError` idiom into `openForm`/`expect` combinators; keyword→parser table for `parseExpr`.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 20/25; large internal churn in the hottest file. The `(void)` expression-parse gap it exposes is a documented follow-up, not part of the extraction.

## bind-result-helper

- **Status**: proposed
- **Score**: 19/25 (leverage 3, locality 4, blast radius 1, heat 4)
- **Files**: ~1-3 estimated
- **Modules**: `src/Interpreter.cpp` (optionally `src/include/AST.h`, `src/include/ASTRuntime.h`)
- **Summary**: Extract one `bindResult` helper for multiple-values destructuring, used by let/letrec/define, eliminating the duplicated 1-id/N-id logic with divergent error text (`bindValues` at Interpreter.cpp:54 vs the inline `Frame::Define` arm at :303, which does not call it).
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 19/25; smaller leverage than the pick. 2026-09-10: friction grown — the two arms now also diverge on the `Value` seam (`bindValues` passes/borrows a `Value`; the `Define` arm `takeLegacy()`s then re-wraps); overlaps `value-register-take-vs-borrow`.

## environment-deepen

- **Status**: proposed
- **Score**: 19/25 (leverage 4, locality 4, blast radius 2, heat 3)
- **Files**: ~4 estimated
- **Modules**: `src/include/Environment.h`, `src/Environment.cpp`, `src/AST.cpp`, `src/Interpreter.cpp`
- **Summary**: Fold Environment + Scope + free functions + interpreter-owned cycle-breaking (`AllScopes`) into one scope module with `contains()`, arena ownership, and pointer-identity keys; delete dead surface.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 19/25 (heat 2→3 after 2026-09-10 re-score); still below the pick. #195 added a third concern (`toShared()`/`Value::share`) to the Environment/Scope tangle and `envSet` still walks the chain twice (`lookup`+`add`). Borderline deletion test: the win depends on the arena absorbing teardown, not merely moving the free functions onto methods.

## visitor-defaults-dead-code

- **Status**: proposed
- **Score**: 16/25 (leverage 3, locality 3, blast radius 2, heat 3)
- **Files**: ~3-5 estimated
- **Modules**: `src/include/ASTVisitor.h`, `src/AnalysisFreeVars.{cpp,h}`, `src/Interpreter.cpp`, CMake
- **Summary**: Default `ASTVisitor`'s 29 pure virtuals to no-ops and remove the entirely-dead `AnalysisFreeVars` pass and undefined `Lambda::findFreeVariables`; optionally collapse the 17 byte-identical `deliver(clone())` self-quoting visit overrides (`Interpreter.cpp:776-842`) behind one hook.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 16/25; the dead-code half is a cleanup, not a deepening; lowest leverage.
- **Reconciled 2026-09-11**: self-quoting `deliver(clone())` overrides now 14, not 17 — #199/#201 turned `BooleanLiteral`/`Char`/`Void` into immediates. The dead `AnalysisFreeVars` core is fully present and latently buggy; score unchanged.

## clone-unique-wrap

- **Status**: proposed
- **Score**: 19/25 (leverage 4, locality 3, blast radius 2, heat 4)
- **Files**: ~5 estimated
- **Modules**: `src/include/AST.h` (`ClonableNode` template), 26 wrap-sites across `src/Interpreter.cpp`, `src/Runtime.cpp`, `src/ASTRuntime.cpp`, `src/AST.cpp`
- **Summary**: `ASTNode::clone()` returns a raw owning `ASTNode*`; 26 callers re-wrap it into `unique_ptr<...>(x->clone())`, an identical leak-prone ceremony. Add a non-virtual `clonePtr()` on `ClonableNode` (or a free `cloneUnique`) that wraps once — the virtual must stay covariant-raw since `unique_ptr` is not covariant — mirroring `Formal::clone()`, which already returns `unique_ptr<Formal>`.
- **First seen**: 2026-09-11
- **Reason (deprioritised)**: 19/25, below the 2026-09-11 pick (22/25); mostly mechanical churn across 5 files.

## range-adapter-dedup

- **Status**: proposed
- **Score**: 19/25 (leverage 3, locality 4, blast radius 1, heat 4)
- **Files**: ~1-2 estimated
- **Modules**: `src/include/AST.h`
- **Summary**: Five hand-rolled range adapters — three byte-identical `IdRange` classes (`DefineValues`, `ListFormal`, `LetValues`) plus `Linklet::FormRange` and `Values::ExprRange` — each an interface as large as its body. Collapse to one generic `IterRange<T>` / `std::ranges::subrange`; the code's own FIXMEs (AST.h:534, :784) already ask for a `view_interface`.
- **First seen**: 2026-09-11
- **Reason (deprioritised)**: 19/25, below the pick; callers keep the same range-for usage, only the definitions collapse.

## operand-accumulate-seam

- **Status**: proposed
- **Score**: 18/25 (leverage 3, locality 2, blast radius 1, heat 5)
- **Files**: ~1 estimated
- **Modules**: `src/Interpreter.cpp`
- **Summary**: `step(App)`, `step(MkValues)` and `step(LetBind)` share a byte-identical accumulate-then-advance operand prologue (Interpreter.cpp:207-251); only the "all done" epilogue differs. An `advanceOrFinish(K)` seam owns the operand state machine, leaving the epilogue per-arm.
- **First seen**: 2026-09-11
- **Reason (deprioritised)**: 18/25, below the pick; the state machine is already localised in the `Frame` types, so locality gain is modest.

## abort-eval-fail-helper

- **Status**: dropped
- **Score**: 15/25 (leverage 2, locality 2, blast radius 1, heat 4)
- **Files**: ~1 estimated
- **Modules**: `src/Interpreter.cpp`
- **Summary**: The `Diag.error(loc, msg); abortEval(); return;` epilogue repeats at ~11 sites; a `return fail(loc, msg)` helper would collapse the 3 lines.
- **First seen**: 2026-09-11
- **Reason (dropped)**: Leverage 2 — the interface barely deepens and callers do the same work; a cosmetic epilogue collapse, not a deepening. Recorded so the next run does not re-derive it.
