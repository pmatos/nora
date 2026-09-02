# Architecture deepening backlog

Persistent memory for the `pm-deepen` routine. One `## <slug>` entry per candidate ever seen; statuses change, rows stay. `### Run` blocks under an entry (or under `## Run log`) are history, not candidates.

## value-printing-raw-ostream-seam

- **Status**: proposed
- **Score**: 22/25 (leverage 5, locality 4, blast radius 2, heat 4)
- **Files**: ~5-6 estimated
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/include/ASTRuntime.h`, `src/ASTRuntime.cpp`, `src/main.cpp`
- **Summary**: Route every `ValueNode::write` through one injected `llvm::raw_ostream` seam and a single `isSelfQuoting` predicate, retiring the three unsynchronised output channels (`std::cout`, `llvm::outs`, `gmp_printf`).
- **First seen**: 2026-09-02
- **Picked**: this run (2026-09-02); flipped to `in-flight` with the PR number at step 6.

## frame-per-kind-continuation

- **Status**: proposed
- **Score**: 22/25 (leverage 5, locality 5, blast radius 3, heat 4)
- **Files**: ~2 estimated (but ~500 lines, the whole CEK loop)
- **Modules**: `src/include/Interpreter.h`, `src/Interpreter.cpp`
- **Summary**: Split the fat multi-purpose `Frame` struct into one small per-`Kind` continuation type with a `resume()` transition, replacing the 13-arm `continueStep` switch with dispatch.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: Ties the pick at 22/25 but loses the deterministic tie-break on blast radius (3 vs 2); highest-risk change in the tree, wants characterization tests first. Runner-up candidate.

## formal-deep-interface

- **Status**: proposed
- **Score**: 21/25 (leverage 4, locality 4, blast radius 1, heat 4)
- **Files**: ~3 estimated
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/Interpreter.cpp`
- **Summary**: Give the shallow `Formal` tag a deep interface (`accepts`, argument-binding, `boundVars`) so arity, binding, and dump stop re-dispatching on `Formal::Type` at four sites; also fixes the per-application formals-copy bug.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 21/25, within one point of the pick; the safest textbook shallow→deep refactor and the natural next firing.

## parse-form-combinators

- **Status**: proposed
- **Score**: 20/25 (leverage 4, locality 4, blast radius 2, heat 4)
- **Files**: ~2 estimated (~10-15 functions in the largest file)
- **Modules**: `src/Parse.cpp`, `src/include/Parse.h`
- **Summary**: Fold the repeated `getPosition/gettok/rewind` form prologue and the `if (!hadError) parseError` idiom into `openForm`/`expect` combinators; keyword→parser table for `parseExpr`.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 20/25; large internal churn in the hottest file. The `(void)` expression-parse gap it exposes is a documented follow-up, not part of the extraction.

## bind-result-helper

- **Status**: proposed
- **Score**: 19/25 (leverage 3, locality 4, blast radius 1, heat 4)
- **Files**: ~1-3 estimated
- **Modules**: `src/Interpreter.cpp` (optionally `src/include/AST.h`, `src/include/ASTRuntime.h`)
- **Summary**: Extract one `bindResult` helper for multiple-values destructuring, used by let/letrec/define, eliminating the duplicated 1-id/N-id logic with divergent error text.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 19/25; smaller leverage than the pick.

## environment-deepen

- **Status**: proposed
- **Score**: 18/25 (leverage 4, locality 4, blast radius 2, heat 2)
- **Files**: ~4 estimated
- **Modules**: `src/include/Environment.h`, `src/Environment.cpp`, `src/AST.cpp`, `src/Interpreter.cpp`
- **Summary**: Fold Environment + Scope + free functions + interpreter-owned cycle-breaking into one scope module with `contains()`, arena ownership, and pointer-identity keys; delete dead surface.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 18/25; `Environment.cpp` is cold (heat 2), so YAGNI docks the pick despite real inefficiencies.

## visitor-defaults-dead-code

- **Status**: proposed
- **Score**: 16/25 (leverage 3, locality 3, blast radius 2, heat 3)
- **Files**: ~3-5 estimated
- **Modules**: `src/include/ASTVisitor.h`, `src/AnalysisFreeVars.{cpp,h}`, `src/Interpreter.cpp`, CMake
- **Summary**: Default `ASTVisitor`'s 27 pure virtuals to no-ops and remove the entirely-dead `AnalysisFreeVars` pass and undefined `Lambda::findFreeVariables`.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: 16/25; the dead-code half is a cleanup, not a deepening; lowest leverage.
