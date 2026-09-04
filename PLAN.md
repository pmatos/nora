# PLAN — issue #92: M0-N Global-consistent alpha/gensym normalizer

## 1. Problem restated

The M0 oracle harness (`test/oracle/`, landed in #193) diffs `norac`'s printed
result against Racket's for a handful of scalar-valued fixtures, piped
through a `%normalize` stage that today only trims whitespace
(`scripts/oracle-normalize.py`). M10 needs something much stronger: a
canonicalizer that can diff two **fully-expanded programs** (`syntax->datum`
of `expand` output) and call them equal modulo (a) every bound identifier's
exact spelling — user-written temporaries, `let-values` desugaring
temporaries, and macro-lifted definitions all get expander-invented or
expander-shifted names — and (b) the numeric suffix on any gensym'd symbol
that survives into quoted data, since that counter's absolute value depends
on unrelated prior allocations (R7) and is never the same twice. The
acceptance bar set by the issue and by M10's own acceptance criterion is
**Racket-vs-Racket**, not norac-vs-Racket: two `expand` runs of the pinned
Racket over equivalent input must normalize to identical text. norac has no
`expand`/syntax-object/module-path-index machinery at all yet (it is a
6-primitive linklet-body interpreter), so module-path-index and
scope-annotated-identifier handling — both syntax-object-level concerns —
are out of reach until M8/M9 land real data. This slice builds and proves
the normalizer's core algorithm now, against real `racket -v 9.3` `expand`
output (the same pinned version as `ORACLE_RACKET_COMMIT`), entirely in the
oracle-harness tooling layer, so M10 inherits a proven mechanism instead of
starting from the current whitespace-only stub.

## 2. Files to touch

All changes are in the oracle-harness tooling layer; **no `src/` or
`src/include/` production-code changes**. See §4 for why.

- `scripts/oracle_datum.py` — **new.** S-expression reader/writer for the
  subset of Racket's `write` grammar the harness needs (symbols, numbers,
  booleans, strings, chars, proper/dotted pairs, vectors). Confirmed
  empirically (`(write '(quote x))` → `(quote x)`, `(write ''(1 2 .
  the-end))` → `(quote (1 2 . the-end))`) that plain `write` — which is what
  `oracle-expand.rkt` uses on `syntax->datum` output — never emits reader
  shorthand (`'`/`` ` ``/`,`/`,@`); `quote`/`quasiquote`/`unquote`/
  `unquote-splicing` always appear as ordinary list forms. So the parser
  does **not** need reader-shorthand support — one less grammar corner, and
  the slice-1 round-trip test uses `"(quote foo)"`, not `"'foo"`.
- `scripts/oracle_alpha.py` — **new.** The binder-driven alpha-renaming
  walker over the fully-expanded-program grammar, plus the quoted-data
  gensym-renumbering pass. This is the actual normalizer algorithm.
- `scripts/oracle-normalize.py` — **modify.** Replace the whitespace-only
  stub's body with: parse → (if the datum is a `module` form) run the
  `oracle_alpha` normalizer, else pass the datum through unchanged → print.
  Non-module input (today's scalar fixtures: `2`, `(1 2 3)`, …) must keep
  normalizing exactly as before so `arithplus.rkt`/`arithmul.rkt`/`box.rkt`/
  `closure.rkt`/`pair.rkt` don't regress.
- `scripts/oracle-expand.rkt` — **new.** `%racket-expand` driver: reads a
  module s-expression fixture, `(parameterize ([current-namespace
  (make-base-namespace)]) (expand datum))`, `write`s
  `(syntax->datum expanded)` + newline. Fresh namespace per process —
  confirmed empirically that this makes `expand`'s internal gensym
  allocations reproduce identically across separate process runs of a
  *given* fixture (two independent process invocations of the same input
  printed the same `lifted/15`/`g332`-style names). Takes an optional
  second CLI argument, a warmup count `N`; when present, calls `(gensym)`
  N times before `expand` runs, precisely shifting the process's gensym
  counter baseline (verified: N=7 shifts a quoted-gensym result from
  `g332` to `g339`, i.e. +7 exactly) — this is the controlled perturbation
  slice 9 uses to prove the normalizer tolerates a shifted counter.
- `scripts/oracle-observe.rkt` — **new.** The observational-equivalence
  fallback driver: `compile-linklet`/`instantiate-linklet`-equivalent
  evaluation of the *expanded* module (via `dynamic-require` on the
  in-memory expanded form, or `eval` in a fresh namespace) that prints the
  module's observable output, for fixtures where structural normalization
  provably cannot reconcile two hygienically-distinct same-named binders
  (see §5, item 3).
- `scripts/test_oracle_normalize.py` — **new.** stdlib `unittest` (not
  pytest — CI's oracle job only `pip install`s `lit`, and this must run
  without Racket installed too, so it belongs in the default `ctest`
  matrix, not the opt-in oracle job). Exercises `oracle_datum` and
  `oracle_alpha` directly on hand-crafted datums; no `racket` binary
  required.
- `test/oracle/CMakeLists.txt` — **modify.** Add
  `add_test(NAME oracle_normalize_unittests COMMAND ${Python3_EXECUTABLE}
  -m unittest discover -s ${CMAKE_SOURCE_DIR}/scripts -p 'test_oracle_*.py')`
  (or equivalent) so the new unit tests run under the default `debug`
  preset without requiring `NORA_HAVE_RACKET`.
- `test/oracle/lit.cfg.py` — **modify.** Register `%racket-expand`
  (→ `oracle-expand.rkt` with no warmup arg), `%racket-expand-warmup7`
  (→ `oracle-expand.rkt` + `7`), and `%observe` (→ `oracle-observe.rkt`)
  substitutions alongside the existing `%racket`/`%normalize`.
- `test/oracle/expand-gensym-shift.rkt`, `test/oracle/expand-shadow.rkt` —
  **new** lit fixtures (`REQUIRES: racket`), see slices 9–10 below.

## 3. TDD slices

Each slice is one commit (squash merges are disabled on this repo, so these
stay as the reviewable history).

1. **Datum round-trip.** Test (`test_oracle_normalize.py::DatumRoundTrip`):
   parsing `"(1 2 . the-end)"`, `"(quote foo)"`, `"#t"`, `"\"a str\""`,
   `"#\\a"`, `"#(1 2 3)"`, `"(define-values (x) 1)"` and re-printing yields
   the original text (whitespace-normalized). Production:
   `oracle_datum.parse(text) -> Datum`, `oracle_datum.write(datum) -> str`.
   Also creates `test/oracle/CMakeLists.txt`'s new `add_test` in this same
   commit so the suite is wired into `ctest --preset debug` from the start.

2. **`%racket-expand` plumbing, no renaming yet.** Test
   (`test/oracle/expand-noop.rkt`, `REQUIRES: racket`): a module with no
   macros/lifts/gensyms (`(module m racket/base (define-values (x) 1) x)`)
   run through `%racket-expand | %normalize` twice produces byte-identical
   output. `%normalize` at this point still only does the old
   whitespace-trim (module-aware branch not yet implemented) — this proves
   the new substitutions and `oracle-expand.rkt` process wiring work before
   any normalizer logic exists.

3. **Binder renaming: `define-values` + `lambda` + `#%app`/`if`/`quote`.**
   Test: two hand-built datums identical except one `define-values` name
   (`lifted/15` vs `lifted/9`) and one `lambda` parameter name normalize to
   the same output. Production: `oracle_alpha.normalize(datum) -> Datum`
   walks `module`/`#%module-begin`/`define-values`/`lambda`/`#%app`/`if`/
   `quote`, threading a `dict[str, str]` env, renaming every binder
   occurrence (parameter, `define-values` LHS) to `v0, v1, …` in
   first-binder-appearance order and rewriting every bound reference through
   the env — an unbound reference (global/primitive) passes through
   unchanged.

4. **`let-values`/`letrec-values` binders.** Test: two datums differing only
   in `let-values` temp names (the `let*`/lift-desugaring shape from
   probe4/probe2) normalize identically. Production: extend the walker's
   grammar table with `let-values`/`letrec-values`, reusing the same
   binder-renaming helper as slice 3.

5. **Remaining grammar: `case-lambda`, `set!`, `begin`/`begin0`/
   `with-continuation-mark`, `#%top`, `#%variable-reference`,
   `#%expression`.** Test: a `case-lambda` with two clauses of different
   arities normalizes clause-by-clause (each clause's parameter list is its
   own binder scope); `set!` renames its target through the env like any
   other reference, not as a binder. Production: extend the grammar table;
   forms with no binding power just recurse into every subform.

6. **`#%provide`/`#%require` boundary.** Test: a `define-values`-bound name
   that is also named in a `#%provide` clause keeps its **original** text
   in the output (both at the definition site and the provide clause),
   while an unexported `define-values` name is still renamed to `vN`; a
   `#%require` module-path datum (`racket/list`, `(for-syntax racket/base)`)
   passes through completely untouched. Production: a first pass collects
   the set of provided names before the renaming walk; the walk special-cases
   `#%provide`/`#%require` as opaque (recurse for well-formedness only, no
   renaming) and excludes provided names from the `define-values` rename.

7. **Quoted-data gensym pass.** Test: `'g574` vs `'g12` inside a `quote`
   subtree (not a binder position) normalize identically via first-occurrence
   renumbering scoped to `quote`/`quote-syntax` data; a quoted symbol with a
   trailing digit that is **not** gensym-shaped, e.g. `'utf-8` or `'sha256`,
   must be **left untouched**. Confirmed empirically that a naive
   trailing-digit regex (`...[0-9]+$`) false-positives on both of those —
   `sha256` and `utf-8` both print as plain symbols indistinguishable in
   text from a gensym — so the pattern must anchor on Racket's actual
   default gensym prefixes (bare `"g"`, per both Racket's and norac's
   `(gensym)` default base, and `"lifted/"`, the expander's lift-name
   prefix seen in probe2/probe6), not "any identifier ending in digits."
   Production: a small second traversal restricted to `quote`/`quote-syntax`
   payloads, matching `^(g|lifted/)[0-9]+$` and renumbering by first
   occurrence within that subtree. Document as an accepted heuristic gap
   (§5): a fixture that quotes the literal data `'g42` for unrelated reasons
   would still be misnormalized — plain `write` output cannot distinguish
   an interned symbol from an uninterned gensym'd one (confirmed: `(write
   (gensym "g"))` prints identically to any interned symbol of that name),
   so this is a fundamental ambiguity in text-only normalization, not a
   solvable bug; anchoring on real generator prefixes only shrinks the
   collision surface.

8. **Wire into `%normalize`.** Test: `test/oracle/expand-noop.rkt` (slice 2)
   still passes, and the five pre-existing scalar oracle fixtures
   (`arithplus.rkt`, `arithmul.rkt`, `box.rkt`, `closure.rkt`, `pair.rkt`)
   still pass unmodified — proving the module-detection branch in
   `oracle-normalize.py` doesn't disturb non-module scalar normalization.
   Production: `oracle-normalize.py`'s `main()` parses stdin via
   `oracle_datum.parse`, dispatches to `oracle_alpha.normalize` only when
   the top-level datum is a `(module …)` form, else keeps the old
   line-based whitespace behavior, then writes the result.

9. **Adversarial gensym-shift fixture.** Verified empirically which
   mechanism actually shifts under a warmup perturbation, since not all of
   them do: expanding a module using `syntax-local-lift-expression` twice in
   the same namespace (with an unrelated module expanded in between) leaves
   its `lifted/N` counter **unchanged** (stayed `lifted/15` both times) — so
   that mechanism cannot demonstrate a counter shift and slice 9 does not
   use it for that purpose. A macro that embeds a literal `(gensym)` result
   as quoted data, by contrast, shifts precisely and reproducibly: two
   fresh-process runs of the same fixture with no perturbation both print
   `(quote g332)`; adding exactly 7 throwaway `(gensym)` calls before
   `expand` runs shifts it to `(quote g339)` — the +7 lands exactly on the
   counter, confirming this is deterministic, controllable perturbation,
   not noise. Test: `test/oracle/expand-gensym-shift.rkt`
   (`REQUIRES: racket`) — a module whose macro does `(with-syntax ([s
   (gensym)]) #''s)` (embedding the gensym as quoted data, forcing slice 7's
   renumbering pass to fire), driven by `oracle-expand.rkt` extended to take
   an optional warmup-count argument (`%racket-expand` vs. a second
   substitution `%racket-expand-warmup7` that calls `(gensym)` 7 times
   before `expand`); two RUN lines invoke the two substitutions on the same
   fixture, pipe both through `%normalize`, and `diff`. This is the direct
   empirical proof of the issue's acceptance criterion ("tolerant of a
   shifted gensym counter … two runs … normalize identically").

10. **Shadowing fixture.** Test: `test/oracle/expand-shadow.rkt` — a module
    whose `let-values` binds two lexically-distinct locals via a
    macro-generated pattern that happens to reuse a name (proving the
    global-consistent, not local, renaming requirement: the same source
    text `t` bound twice in different non-nested scopes must resolve to two
    *different* canonical names, `vN`/`vM`, not collide). If a genuine
    duplicate-binder-in-one-frame case can be constructed (the hygiene hole
    from §5 item 3, e.g. via a `for`-loop expansion introducing two
    same-printed-name temporaries in one `let-values` clause list), route
    that fixture instead through `%racket-expand | %observe` (using
    `oracle-observe.rkt` from this same slice) and document it as the
    fallback's motivating case rather than trying to make the structural
    normalizer handle it.

## 4. AST/visitor surface

**N/A.** No `ASTNode` kind is added or changed; no `ASTVisitor`/`Interpreter`
`visit()` overload is touched. The normalizer operates entirely on Racket
`write`-format text produced by an external `racket` process
(`oracle-expand.rkt`) and consumed by a Python script — norac's C++ AST is
untouched because norac has no `expand` mode and no syntax-object
representation to normalize in the first place.

## 5. Risk areas

- **`write()` Racket-compatibility must not be touched.** It would be
  tempting to make normalization easier by having norac's `write()` emit a
  marker for uninterned symbols (e.g. `g574#uninterned`) — **do not do
  this**. `write()`'s output is asserted byte-for-byte against Racket's own
  `write` by every `CHECK:` line in `test/integration/`; any divergence
  breaks that contract. The normalizer must work from unmarked printed text,
  which is exactly why it needs the grammar-aware binder walk (§3 slices
  3–6) rather than a marker-assisted rewrite.
- **Free-variable analysis / memory safety: not applicable.** This slice
  makes zero changes under `src/`, so `AnalysisFreeVars` is untouched and no
  `asan`/`ubsan` preset run is needed for this PR. (Flag this explicitly in
  the PR description so reviewers don't go looking for an asan run that
  doesn't apply.)
- **Hygiene loss in `syntax->datum`.** Confirmed empirically (probe2/probe6):
  `syntax->datum` strips all scope-set information, so two hygienically
  distinct identifiers that print with the same text and are bound in the
  *same* binder frame (not nested) are structurally indistinguishable to a
  pure-datum walker. This is a genuine, documented blind spot, not an
  oversight — slice 10 either demonstrates it's unreachable for the grammar
  subset in scope, or routes the one case that reaches it to the
  observational-equivalence fallback (`oracle-observe.rkt`). Do not attempt
  to "solve" this in this slice; a real fix needs Racket-side
  `bound-identifier=?`-driven pre-renaming, filed as follow-up (§6).
- **Heuristic scope creep, and a residual false-positive gap that cannot be
  fully closed.** The quoted-data gensym pattern (slice 7) is deliberately
  narrow — `^(g|lifted/)[0-9]+$`, only inside `quote`/`quote-syntax`
  subtrees, never applied to binder/reference identifiers — specifically
  because a generic "any trailing-digit identifier" pattern was verified to
  false-positive on ordinary data (`sha256`, `utf-8` both print
  indistinguishably from a gensym'd `gNN` symbol; plain `write` carries no
  interned/uninterned marker). Anchoring on real generator prefixes shrinks
  but does not eliminate the collision surface — a fixture that
  legitimately quotes `'g42` would still be misnormalized. This is a
  documented, accepted limitation of text-only normalization, not something
  to "fix" by further pattern tuning; resist the urge to generalize the
  pattern in the other direction either.
- **`ORACLE_RACKET_COMMIT` pin.** All new fixtures assume the installed
  `racket` matches the pinned v9.3 build (already true in this workspace).
  If the pin is ever bumped, re-run slices 9–10's fixtures to confirm the
  lift-naming convention (`lifted/N`) and gensym base (`"g"`) haven't
  changed between Racket versions — nothing in this plan hardcodes that
  assumption defensively, so a silent version drift would show up as a
  fixture regression, which is the intended fail-closed behavior.

## 6. Out of scope

- **Module-path-index encoding and scope-annotated identifiers.** Verified
  empirically (probe3): `syntax->datum` of a simple `#%require` doesn't
  expose module-path-index structure at all — it stays a plain module-path
  datum. There is nothing to normalize yet; this becomes real work once
  M11+ gives NORA (and this harness) an actual module-instantiation path
  that surfaces MPIs. Filed as a follow-up against M10, not this issue.
- **norac-side `expand`/syntax-object support.** Out of reach until M8/M9;
  this plan proves the algorithm Racket-vs-Racket so M10 has a working
  normalizer to point at norac's future `expand` output rather than
  building it from scratch under M10's own time pressure.
- **Hygiene-aware pre-renaming on the Racket side** (using
  `bound-identifier=?` before `syntax->datum` strips scope info), which
  would close the shadowing blind spot from §5 permanently instead of
  routing it to the observational fallback. Real improvement, but a
  distinct chunk of work with its own design questions (how to serialize a
  hygiene-aware rename back through plain datum text); filed as a follow-up.
- **CI workflow changes.** `oracle.yml` stays `workflow_dispatch`-only per
  its existing comment; this plan's new unit tests (slice 1) run in the
  default `debug` preset without Racket, and the new lit fixtures
  (`REQUIRES: racket`) only execute when a developer runs `ctest -R oracle`
  locally with Racket installed, exactly like the five existing oracle
  fixtures. No `.github/workflows/*.yml` edits.
- **norac's `GensymFunction`/`IdPool`.** No change needed — norac's own
  gensym counter is already per-process-deterministic (`static unsigned
  Counter` in `src/Runtime.cpp`), and this slice's acceptance target is
  Racket-vs-Racket, not norac-vs-Racket, so a norac-vs-Racket oracle fixture
  using `(gensym)` is left as an easy follow-up once this normalizer is
  proven, not a deliverable here.
- **Refactoring `scripts/oracle-eval.rkt` or the existing 5 oracle
  fixtures.** They stay exactly as M0 left them; slice 8 only adds a branch
  to `oracle-normalize.py`, verified by re-running them unmodified.
