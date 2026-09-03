# Architecture deepening backlog

Persistent memory for the `pm-deepen` routine. One `## <slug>` entry per candidate ever seen; statuses change, rows stay. `### Run` blocks under an entry (or under `## Run log`) are history, not candidates.

## value-printing-raw-ostream-seam

- **Status**: landed
- **Score**: 22/25 (leverage 5, locality 4, blast radius 2, heat 4)
- **Files**: ~5-6 estimated (actual: 6 — the five modules plus `test/unit/test_parse.cpp`)
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/include/ASTRuntime.h`, `src/ASTRuntime.cpp`, `src/main.cpp`
- **Summary**: Route every `ValueNode::write` through one injected `llvm::raw_ostream` seam and a single self-quoting `isa<>` set, retiring the three unsynchronised output channels (`std::cout`, `llvm::outs`, `gmp_printf`).
- **First seen**: 2026-09-02
- **PR**: #141 (branch `sym/nora/routine/refactor-audit/01M1GPA0JP`, adopted) — **merged 2026-09-03**

## frame-per-kind-continuation

- **Status**: in-flight
- **Score**: 22/25 (leverage 5, locality 5, blast radius 3, heat 4)
- **Files**: ~3 estimated (`src/include/Interpreter.h`, `src/Interpreter.cpp`, `test/unit/test_interpreter.cpp`) — the CEK loop is ~2 source files but ~470 lines; the prior ~2 estimate omitted the test file step 5 must edit
- **Modules**: `src/include/Interpreter.h`, `src/Interpreter.cpp`
- **Summary**: Split the fat multi-purpose `Frame` struct (13-value `Kind` enum, ~20 kind-specific fields) into one small per-`Kind` continuation type with a `resume()` transition, replacing the 13-arm `continueStep` switch with dispatch, while preserving the GC-scanned `Kont` buffer, the universal `Marks` header, and the Call/WcmMark/Halt tail-call reuse seam.
- **First seen**: 2026-09-02
- **Reason (picked)**: Top surviving `proposed` candidate at 22/25 once `value-printing-raw-ostream-seam` landed. Within 1 point of the runner-up candidate `formal-deep-interface` (21/25).

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
- **Reason (deprioritised)**: 19/25; smaller leverage than the pick.

## environment-deepen

- **Status**: proposed
- **Score**: 18/25 (leverage 4, locality 4, blast radius 2, heat 2)
- **Files**: ~4 estimated
- **Modules**: `src/include/Environment.h`, `src/Environment.cpp`, `src/AST.cpp`, `src/Interpreter.cpp`
- **Summary**: Fold Environment + Scope + free functions + interpreter-owned cycle-breaking (`AllScopes`) into one scope module with `contains()`, arena ownership, and pointer-identity keys; delete dead surface.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 18/25; `Environment.cpp` is cold (heat 2), so YAGNI docks the pick despite real inefficiencies. Borderline deletion test: the win depends on the arena absorbing teardown, not merely moving the free functions onto methods.

## visitor-defaults-dead-code

- **Status**: proposed
- **Score**: 16/25 (leverage 3, locality 3, blast radius 2, heat 3)
- **Files**: ~3-5 estimated
- **Modules**: `src/include/ASTVisitor.h`, `src/AnalysisFreeVars.{cpp,h}`, `src/Interpreter.cpp`, CMake
- **Summary**: Default `ASTVisitor`'s 29 pure virtuals to no-ops and remove the entirely-dead `AnalysisFreeVars` pass and undefined `Lambda::findFreeVariables`; optionally collapse the 17 byte-identical `deliver(clone())` self-quoting visit overrides (`Interpreter.cpp:776-842`) behind one hook.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 16/25; the dead-code half is a cleanup, not a deepening; lowest leverage.

## Run log

### Run 2026-09-04 — complete

- **Outcome**: complete
- **Stopped at**: step 6 — PR opened for `frame-per-kind-continuation`
- **Branch**: `sym/nora/routine/refactor-audit/01M1MR2N1J` (adopted — non-default, no unique history, no upstream, unpublished on origin; kept the caller's name, not renamed)
- **Committed**: report `.architecture/reviews/2026-09-04-frame-per-kind-continuation.md`, reconciled backlog, and the implementation
- **Evidence**: `value-printing-raw-ostream-seam` PR #141 reconciled MERGED → landed; no open architecture PR blocking; toolchain (LLVM 22 / GMP / libgc / Catch2) verified by `cmake --preset release`
- **Next**: review the PR; the natural next firing is `formal-deep-interface` (21/25)
