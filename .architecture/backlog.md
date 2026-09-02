# Architecture deepening backlog

Persistent memory for the `pm-deepen` routine. One `## <slug>` entry per candidate
ever surfaced; statuses change, rows stay. `### Run` blocks under the *Run log*
heading are firing history, not candidates.

> **Reconciliation note (2026-09-03).** `.architecture/backlog.md` does not yet
> exist on `origin/main`: the prior firing's copy lives on the unmerged branch of
> **PR #141**. This firing merged that backlog with its own independent scan,
> **reusing the prior firing's slugs** for equivalent candidates so dedup keeps
> working. Slug aliases from this firing's scan: `formal-variant-tag-dispatch` →
> `formal-deep-interface`; `environment-scope-shallow-wrapper` →
> `environment-deepen`; `visitor-leaf-noop-boilerplate` → `visitor-defaults-dead-code`;
> `parse-form-combinator` → `parse-form-combinators`. Scores for shared candidates
> are the prior firing's established values (the incumbent memory); divergences from
> this firing's independent read are noted inline. Ranked, the highest-scored
> **proposed** candidate in the merged list is `frame-per-kind-continuation` (22/25).

## value-printing-raw-ostream-seam

- **Status**: in-flight
- **Score**: 22/25 (leverage 5, locality 4, blast radius 2, heat 4)
- **Files**: 6 (five modules + `test/unit/test_parse.cpp`)
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/include/ASTRuntime.h`, `src/ASTRuntime.cpp`, `src/main.cpp`
- **Summary**: route every `ValueNode::write` through one injected `llvm::raw_ostream`
  seam and a single self-quoting `isa<>` set, retiring the three unsynchronised
  output channels (`std::cout`, `llvm::outs`, `gmp_printf`).
- **First seen**: 2026-09-02
- **PR**: #141 (branch `sym/nora/routine/refactor-audit/01M1GPA0JP`, adopted, OPEN, mergeable)
- **Reason**: open architecture PR — blocks new implementation under "one
  architecture PR at a time" until it merges or closes.

## frame-per-kind-continuation

- **Status**: proposed
- **Score**: 22/25 (leverage 5, locality 5, blast radius 3, heat 4)
- **Files**: ~2 estimated (but ~500 lines — the whole CEK loop)
- **Modules**: `src/include/Interpreter.h`, `src/Interpreter.cpp`
- **Summary**: split the fat multi-purpose `Frame` struct into one small per-`Kind`
  continuation type with a `resume()` transition, replacing the 13-arm
  `continueStep` switch with dispatch.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: highest-scored proposed candidate but loses the
  deterministic tie-break on blast radius (3); highest-risk change in the tree,
  wants characterization tests first. Carried from the PR #141 backlog; this
  firing's scan did not independently re-surface it (its passes scoped the reader
  and the runtime/env, not the CEK continuation loop) — kept because reconciliation
  never drops a known candidate.

## formal-deep-interface

- **Status**: proposed
- **Score**: 21/25 (leverage 4, locality 4, blast radius 1, heat 4)
- **Files**: ~3-4 estimated
- **Modules**: `src/include/AST.h`, `src/AST.cpp`, `src/Interpreter.cpp`, `src/AnalysisFreeVars.cpp`
- **Summary**: give the shallow `Formal` tag a deep interface (`accepts`,
  argument-binding, `boundVars`/`dump`) so arity, binding, and dump stop
  re-dispatching on `Formal::Type` at 4-5 sites; also fixes the per-application
  formals-copy bug.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: within one point of the top; the safest textbook
  shallow→deep refactor and a strong next firing. This firing independently
  re-surfaced it as `formal-variant-tag-dispatch` (same candidate, agreed score
  band).

## runtime-builtin-prologue

- **Status**: proposed
- **Score**: 21/25 (leverage 5, locality 4, blast radius 3, heat 4)
- **Files**: ~3-5 estimated
- **Modules**: `src/Runtime.cpp`, `src/include/Runtime.h`, `src/include/ASTRuntime.h`, `src/Interpreter.cpp`
- **Summary**: replace 18 hand-rolled builtin classes with typed primitive
  descriptors + one arity/type-unwrap seam that raises specific diagnostics
  (retiring the generic `"invalid arguments to 'X'"` null channel).
- **First seen**: 2026-09-03
- **Reason (context)**: **new** this firing (not in the PR #141 backlog). Blast
  radius 3 because it touches the builtin argument/return ABI that is mid-migration
  to `nr_value` (`docs/value-model-gc-migration.md`) and the error text pinned by
  `error-*.rkt` integration tests — best revisited after the value-model migration
  settles.

## parse-form-combinators

- **Status**: proposed
- **Score**: 20/25 (leverage 4, locality 4, blast radius 2, heat 4)
- **Files**: ~2-3 estimated (~10-15 functions in the largest file)
- **Modules**: `src/Parse.cpp`, `src/include/Parse.h`, `test/unit/test_parse.cpp`
- **Summary**: fold the repeated `getPosition/gettok/rewind` form prologue and the
  `if (!hadError) parseError` epilogue into an `openForm`/`expect` combinator; leave
  `parseExpr`'s waterfall dispatch unchanged (keyword→parser table is a separate
  follow-up).
- **First seen**: 2026-09-02
- **Reason (context)**: this firing independently re-picked this as its top
  candidate and read blast radius 1 (2-3 files, no published interface) → 21/25;
  the incumbent PR #141 score of 20/25 (blast 2) is retained for continuity. Either
  way it sits inside a 1-point cluster with `formal-deep-interface` and
  `runtime-builtin-prologue`. Implement the combinator as a `static` member template
  of `Parse` (nora CI rejects new file-scope free functions).

## lex-token-cursor-seam

- **Status**: proposed
- **Score**: 20/25 (leverage 5, locality 4, blast radius 3, heat 3)
- **Files**: ~3 estimated (large `Parse.cpp` diff, ~70 call sites)
- **Modules**: `src/include/Lex.h`, `src/Lex.cpp`, `src/Parse.cpp`
- **Summary**: add `peekTok`/`expect`/`consumeIf` so the ~70 hand-rolled
  gettok+`.is`+rewind sites route through one cursor; unifies the two backtrack
  idioms (`rewindTo(Start)` vs `rewind(T.size())`).
- **First seen**: 2026-09-03
- **Reason (context)**: **new** this firing. Overlaps `parse-form-combinators`'
  `expect` idea at the lexer level; `parse-form-combinators` is the smaller first
  step onto this seam.

## bind-result-helper

- **Status**: proposed
- **Score**: 19/25 (leverage 3, locality 4, blast radius 1, heat 4)
- **Files**: ~1-3 estimated
- **Modules**: `src/Interpreter.cpp` (optionally `src/include/AST.h`, `src/include/ASTRuntime.h`)
- **Summary**: extract one `bindResult` helper for multiple-values destructuring,
  used by let/letrec/define, eliminating the duplicated 1-id/N-id logic with
  divergent error text.
- **First seen**: 2026-09-02
- **Reason (context)**: carried from the PR #141 backlog; this firing's scan did not
  independently re-surface it.

## environment-deepen

- **Status**: proposed
- **Score**: 18/25 (leverage 4, locality 4, blast radius 2, heat 2)
- **Files**: ~4 estimated
- **Modules**: `src/include/Environment.h`, `src/Environment.cpp`, `src/AST.cpp`, `src/Interpreter.cpp`
- **Summary**: fold Environment + Scope + free functions + interpreter-owned
  cycle-breaking into one scope module with `contains()`, arena ownership, and
  pointer-identity keys; delete dead surface (`envExtend`).
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: `Environment.cpp` is cold (heat 2), so YAGNI docks it.
  This firing independently re-surfaced it as `environment-scope-shallow-wrapper`
  (same candidate).

## ast-range-view-duplication

- **Status**: proposed
- **Score**: 17/25 (leverage 3, locality 3, blast radius 2, heat 4)
- **Files**: ~2-4 estimated
- **Modules**: `src/include/AST.h`, `src/AST.cpp`
- **Summary**: replace the 5 hand-rolled `begin/end/operator[]` iterator-view
  classes (`FormRange`, two `IdRange`, `LetValues::IdRange`, `Values::ExprRange`)
  with one `NodeRange<T>` (or `llvm::ArrayRef<T>`).
- **First seen**: 2026-09-03
- **Reason (context)**: **new** this firing.

## keyword-token-table-duplication

- **Status**: proposed
- **Score**: 17/25 (leverage 3, locality 4, blast radius 2, heat 3)
- **Files**: ~3 estimated
- **Modules**: `src/Lex.cpp`, `src/include/Lex.h`, `src/Parse.cpp`
- **Summary**: drive lexing, the `isSymbolTok` predicate, and (with the combinator)
  parse dispatch from one keyword table instead of three drifting parallel lists
  (which already disagree: 20 vs 18 entries).
- **First seen**: 2026-09-03
- **Reason (context)**: **new** this firing; pairs with `parse-form-combinators` and
  `lex-token-cursor-seam` as a later reader cleanup.

## visitor-defaults-dead-code

- **Status**: proposed
- **Score**: 16/25 (leverage 3, locality 3, blast radius 2, heat 3)
- **Files**: ~3-5 estimated
- **Modules**: `src/include/ASTVisitor.h`, `src/AnalysisFreeVars.{cpp,h}`, `src/Interpreter.cpp`
- **Summary**: default `ASTVisitor`'s pure virtuals to a no-op / self-evaluating
  hook (removing ~32 near-identical leaf bodies) while keeping "new node ⇒ must be
  handled" for the dispatch visitors; remove dead free-vars surface.
- **First seen**: 2026-09-02
- **Reason (deprioritised)**: lowest leverage; the dead-code half is cleanup, not
  deepening. This firing independently re-surfaced the defaulting half as
  `visitor-leaf-noop-boilerplate`.

## value-handle-passthrough

- **Status**: dropped
- **Score**: leverage 1 (not totalled)
- **Files**: `src/include/Value.h`
- **Modules**: `src/include/Value.h`, `src/Interpreter.cpp`
- **Summary**: `Value` wraps `unique_ptr<ValueNode>` and every consumer immediately
  `takeLegacy()`s it back out.
- **First seen**: 2026-09-03
- **Reason**: Leverage 1 — deletion test *moves*, not concentrates. Intentionally
  transitional per `docs/value-model-gc-migration.md` §3 (becomes a tagged
  `nr_value` word); its shallowness is a planned migration state, not friction.

## formal-clone-boilerplate

- **Status**: dropped
- **Score**: not totalled
- **Files**: `src/include/AST.h:490-575`
- **Modules**: `src/include/AST.h`, `src/AST.cpp`
- **Summary**: the `Formal` family hand-rolls `clone()` three times outside the
  `ClonableNode` CRTP.
- **First seen**: 2026-09-03
- **Reason**: already in the backlog — duplicate of `formal-deep-interface`, which
  reworks the same hierarchy and subsumes it.

## Run log

### Run 2026-09-03 — bailed-preflight (open architecture PR)

- **Outcome**: bailed-preflight — an in-flight architecture PR is open, which the
  contract lists under "Stop before making any change"; the would-be `complete`
  outcome is blocked by the "one PR at a time" rule.
- **Stopped at**: step 2 (reconcile) — a prior firing's architecture PR #141 is
  still open, so a default (implementing) run must not open a second concurrent one.
- **Branch**: `sym/nora/routine/refactor-audit/01M1J5Q8DQ` — **adopted** (non-default,
  0 commits ahead of `origin/main`, no upstream, unpublished on origin). Kept the
  caller's name; adopted branches are never renamed to the slug.
- **Committed**: `.architecture/reviews/2026-09-03-parse-form-combinator.md` and this
  reconciled `.architecture/backlog.md`.
- **Evidence**: open PR **#141** — "refactor(ast): route value printing through one
  injected raw_ostream seam", branch `sym/nora/routine/refactor-audit/01M1GPA0JP`,
  state OPEN, mergeable, created 2026-09-02. Discovered via
  `gh pr list --search pm-deepen`; no `.architecture/backlog.md` exists on
  `origin/main` yet (that backlog lives on #141's unmerged branch), so the standard
  backlog-based dedup could not see it — the `gh` backstop did. This firing's scan
  found 4 candidates new to the backlog (`runtime-builtin-prologue`,
  `lex-token-cursor-seam`, `ast-range-view-duplication`,
  `keyword-token-table-duplication`) and independently re-confirmed four incumbents.
- **Next**: a human merges or closes #141. The next firing then reconciles this
  backlog, re-ranks, and implements the leading proposed candidate test-first. No
  quality gate was run this firing (nothing was implemented), so the shared build
  budget was not spent.
