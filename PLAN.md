# PLAN — Issue #91: M0 — Oracle + differential harness + version lock

## 1. Problem restated

`norac` today has no way to check its interpreter's behavior against real Racket:
every integration test asserts a hand-written `CHECK:` value, so a norac-specific
bug that happens to be internally consistent can never be caught. M0 builds the
first piece of that check — a `test/oracle/` lit harness that runs the same
`(linklet () () ...)` fixture through both the pinned `racket` and `norac`,
normalizes each engine's printed result, and diffs them — gated on a detected
`NORA_HAVE_RACKET` so the suite is a no-op (not a failure) on machines without
Racket. It also introduces the single controlled fact the rest of the roadmap's
version-locking depends on (`ORACLE_RACKET_COMMIT`), wires the recipe that will
someday regenerate `expander.rktl`/the `racket/base` `.zo` graph from that same
commit, adds an opt-in CI job that actually has Racket installed, and adds a
`lsan` CMake preset. This is pure test-infra and build-infra: **no interpreter
(`src/`) behavior changes**, no new AST nodes, no new primitives.

## 2. Files to touch

**New:**
- `ORACLE_RACKET_COMMIT` (repo root) — one line, the pinned upstream
  `racket/racket` commit SHA. Single source of truth for provenance.
- `test/oracle/CMakeLists.txt` — mirrors `test/integration/CMakeLists.txt`:
  `configure_file` the site config + `add_test(NAME oracle COMMAND nora-lit
  .../test/oracle -v)`.
- `test/oracle/lit.cfg.py` — mirrors `test/integration/lit.cfg.py`; adds the
  `%racket` and `%normalize` substitutions and populates
  `config.available_features` with `"racket"` when `NORA_HAVE_RACKET` is true.
- `test/oracle/lit.site.cfg.py.in` — mirrors the integration one; carries
  `@NORA_RACKET_EXECUTABLE@`, `@NORA_HAVE_RACKET@`, `@ORACLE_RACKET_COMMIT@`
  through from CMake.
- `test/oracle/arithplus.rkt`, `arithmul.rkt`, `closure.rkt`, `pair.rkt`,
  `box.rkt` — the "handful" of oracle fixtures (adapted from the
  already-proven `test/integration/` files of the same names).
- `scripts/oracle-eval.rkt` — the `%racket` driver. Reads the `.rkt` file's
  single `(linklet () () form ...)` datum, rewrites it in memory (never
  touches the file on disk) into `(linklet () (oracle-result) form ...
  (define-values (oracle-result) <last-form>))`, compiles it with
  `compile-linklet` and instantiates with `instantiate-linklet` (both from
  `racket/linklet`), then `write`s `(instance-variable-value inst
  'oracle-result)` followed by a newline — the same shape norac's `main.cpp`
  produces (`I.getResult()->write()` + `'\n'`).
- `scripts/oracle-normalize.py` — the `%normalize` filter. For M0 it only
  strips trailing whitespace per line and enforces exactly one trailing
  newline (the current oracle corpus is scalar-only, so nothing more is
  needed yet). This is the scaffold M0-N's alpha/gensym normalizer replaces —
  same script, same substitution name, bigger implementation.
- `.github/workflows/oracle.yml` — the opt-in CI job (see §3 slice 7).

**Modified:**
- `CMakeLists.txt` (root) — `option(NORA_ENABLE_ORACLE "Enable the oracle
  differential test harness" ON)`, `find_program(NORA_RACKET_EXECUTABLE NAMES
  racket)`, derive `NORA_HAVE_RACKET`, read `ORACLE_RACKET_COMMIT` into a CMake
  variable, pass both through to `add_subdirectory(test)`.
- `test/CMakeLists.txt` — add `add_subdirectory(oracle)`.
- `src/CMakeLists.txt` — extend the existing `WITH_UBSAN`/`WITH_ASAN` Debug-only
  guard with `WITH_LSAN`, adding `-fsanitize=leak` compile+link flags.
- `CMakePresets.json` — add an `lsan` configure/build/test preset, mirroring
  `ubsan`.
- `scripts/gen-expander.sh` — default `REF` to the contents of
  `ORACLE_RACKET_COMMIT` when present (falling back to today's
  installed-racket-version heuristic with a warning if the file is missing).
- `ROADMAP.md` — one-line update pointing at the concrete `ORACLE_RACKET_COMMIT`
  file now that it exists (the section already describes the concept).

**Explicitly not touched:** anything under `src/*.cpp`/`src/include/*.h`
(interpreter, AST, visitors), `expander/expander.rktl` (forbidden for this
stage regardless), `src/mlir/`, `src/include/nir/`.

## 3. TDD slices

Each slice is one reviewable commit. "Red" for infra work means: the command
the slice's acceptance check runs currently fails/doesn't exist; "green" means
the new files/config make it pass.

1. **`lsan` CMake preset.**
   Red: `cmake --preset lsan` fails (`Unknown preset "lsan"`).
   Production: add `WITH_LSAN` handling in `src/CMakeLists.txt`
   (`-fsanitize=leak`, Debug-only, alongside the existing ASan/UBSan guard) and
   the `lsan` configure/build/test presets in `CMakePresets.json`.
   Green: `cmake --preset lsan && cmake --build --preset lsan && ctest --preset
   lsan` passes with zero leak reports on the existing ~90-test corpus.
   *Independent of every other slice — safe to land first or in parallel.*

2. **`ORACLE_RACKET_COMMIT` + racket detection, no behavior yet.**
   Red: no such file, no `NORA_HAVE_RACKET` CMake variable exists.
   Production: add the `ORACLE_RACKET_COMMIT` file (pin it to the commit
   backing the currently-installed Racket 9.3 release tag, `v9.3`, as the
   initial value); add `NORA_ENABLE_ORACLE`/`NORA_RACKET_EXECUTABLE`/
   `NORA_HAVE_RACKET` to the root `CMakeLists.txt`; add empty
   `test/oracle/{CMakeLists.txt,lit.cfg.py,lit.site.cfg.py.in}` that configure
   cleanly but list zero tests; wire `add_subdirectory(oracle)` into
   `test/CMakeLists.txt`.
   Green: `cmake --preset debug` configures and logs the detected racket
   path/`NORA_HAVE_RACKET` value; `ctest --preset debug -N` lists a `oracle`
   test entry (even though it currently finds 0 `.rkt` files under it).

3. **First real oracle test: `arithplus.rkt`.**
   Red: add `test/oracle/arithplus.rkt` with `;; RUN: norac %s | %normalize >
   %t.norac`, `;; RUN: %racket %s | %normalize > %t.racket`, `;; RUN: diff
   %t.norac %t.racket` — lit fails immediately because `%racket`/`%normalize`
   aren't registered substitutions.
   Production: `scripts/oracle-eval.rkt`, `scripts/oracle-normalize.py`; wire
   both into `test/oracle/lit.cfg.py` as substitutions (absolute paths, same
   pattern as the existing `not` substitution in `test/integration/lit.cfg.py`).
   Green: `ctest --preset debug -R oracle` passes; manually confirm `racket
   scripts/oracle-eval.rkt test/oracle/arithplus.rkt` prints `2`.

4. **Round out the corpus: `arithmul.rkt`, `closure.rkt`, `pair.rkt`,
   `box.rkt`.**
   Red: four new `.rkt` files added (same RUN-line shape as slice 3); they
   exercise closures/`set!`, mutable pairs, and mutable boxes — none require
   changes to `oracle-eval.rkt` since the "wrap the last form in an export"
   transform is form-agnostic.
   Green: all five oracle tests pass under `ctest --preset debug`.
   *No production code changes expected; if `oracle-eval.rkt` turns out to
   need a fix for one of these shapes, that fix is this slice's production
   code.*

5. **Skip-green-with-no-Racket path.**
   Red: with `-DNORA_ENABLE_ORACLE=OFF` (a deterministic stand-in for "no
   racket on this machine" that doesn't require uninstalling anything), the
   oracle lit tests currently have no way to report "skipped" — lit would
   error trying to substitute `%racket`.
   Production: tag every `test/oracle/*.rkt` with `;; REQUIRES: racket`; have
   `test/oracle/lit.cfg.py` add `"racket"` to `config.available_features` only
   when `NORA_HAVE_RACKET` is true.
   Green: `cmake --preset debug -DNORA_ENABLE_ORACLE=OFF && ctest --preset
   debug` reports the oracle tests as unsupported/skipped, not failed, and the
   overall `ctest` run is green.

6. **`gen-expander.sh` reads the pinned commit.**
   Red: today `gen-expander.sh`'s default `REF` is derived from the *host*
   racket's version tag, independent of any repo-pinned fact.
   Production: default `REF` to `$(cat ORACLE_RACKET_COMMIT)` when that file
   exists (existing explicit-argument override and host-version fallback stay
   for the file-missing case).
   Green: manual verification only (`gen-expander.sh` has no test harness
   today and this slice does not invoke it — see §6 risk). Confirm by reading
   the diff and running `bash -n scripts/gen-expander.sh`.

7. **Opt-in CI `oracle` job.**
   Production: `.github/workflows/oracle.yml`, triggered on
   `workflow_dispatch` only (not `push`/`pull_request` — this is the "opt-in"
   the issue asks for, and it keeps the default CI matrix from gaining a new
   apt dependency). Installs LLVM 22 + build deps (mirroring `test.yml`) plus
   `racket` via apt, configures/builds the `debug` preset, runs `ctest
   --preset debug -R oracle`.
   Green: manually trigger the workflow once after merge and confirm it
   passes; not verifiable inside this sandbox (no CI runner access).

8. **Docs pointer.**
   Production: one-paragraph edit to `ROADMAP.md`'s "Toolchain version lock"
   section naming the concrete `ORACLE_RACKET_COMMIT` file now that it exists.
   No test; pure documentation.

## 4. AST/visitor surface

None. This issue adds no `ASTNode` kind and touches no visitor. No changes to
`ASTNodeKind` ordering, `ASTVisitor.h`, or `Interpreter.h`.

## 5. Risk areas

- **`write()` printed-representation compatibility.** norac's `write()` must
  already match Racket's printed form for the CHECK-based integration suite,
  but known gaps exist (`QuotedExpr::operator==` FIXME; dotted-list/vector
  printing is only lightly tested). Mitigation: the oracle corpus (slice 4) is
  deliberately restricted to fixtures whose result is a plain scalar (fixnum),
  exactly like every existing `test/integration/*.rkt` `CHECK:` target — no
  quoted structures, no printing a pair/box object directly. Extending the
  corpus to compound printed values is follow-up work, not this PR.
- **Racket API assumption, empirically verified but narrow.** The
  `oracle-eval.rkt` design (rewrite last body form into a synthetic export,
  `compile-linklet`/`instantiate-linklet`/`instance-variable-value`) was
  smoke-tested against the Racket 9.3 installed in this environment for
  numeric and closure/mutation bodies and works. It assumes the file's last
  top-level form is an expression (never a `define-values`) — true of every
  test in `test/integration/` today. If a future oracle fixture ends in a
  definition, `oracle-eval.rkt` needs a small extension; flag this as a
  documented assumption in the script rather than silently mishandling it.
- **`lsan` may not start clean.** `docs/value-model-gc-migration.md` already
  notes LSan can misreport Boehm-GC-owned allocations (interned symbols,
  `GC_MALLOC_UNCOLLECTABLE` roots) as leaks, and flags
  `ASAN_OPTIONS=detect_leaks=0` as an escape hatch for the *existing*
  `asan`/`ubsan` presets. The new standalone `lsan` preset (slice 1) should be
  tried clean first; if Boehm's roots trigger false positives, prefer a
  narrow `LSAN_OPTIONS=suppressions=...` file over a blanket
  `detect_leaks=0`, since the latter would make the preset meaningless.
- **CI cost/coupling.** Installing `racket` in the default `test.yml` matrix
  would slow every push/PR and add a new hard apt dependency; keeping the
  oracle job `workflow_dispatch`-only (slice 7) avoids that but also means it
  isn't self-verifying until manually triggered post-merge — call this out in
  the PR description so a reviewer knows to trigger it once.
- **Partial fulfillment of the issue's literal acceptance text.** The issue
  body says artifacts (`expander.rktl`, the `racket/base` `.zo` graph) should
  "carry a recorded provenance stamp matching `ORACLE_RACKET_COMMIT`." This
  stage's constraints forbid editing `expander/expander.rktl`, and the
  `racket/base` `.zo` graph doesn't exist in the repo yet (it's an M16
  deliverable per `ROADMAP.md` line 44). This PR delivers the provenance
  **mechanism** (the pinned commit file, `gen-expander.sh` consuming it) but
  does not run regeneration or embed a stamp into either artifact — that's
  explicitly deferred (see §6). This is a deliberate, documented scope cut,
  not an oversight.

## 6. Out of scope (this PR)

- Actually running `gen-expander.sh` and committing a regenerated
  `expander/expander.rktl` — forbidden by this stage's constraints regardless,
  and a large, independently-reviewable content change on its own merits
  (every downstream parse/primitive-surface doc would need re-validation).
  Filed as explicit follow-up, gated on `ORACLE_RACKET_COMMIT` already existing
  from this PR.
- Building the `racket/base` `.zo` graph / fasl regeneration tooling — this is
  M16 scope per `ROADMAP.md`; M0 only needs the pinned commit to exist so M16
  can consume it.
- The real gensym/alpha-normalizer (M0-N, issue #92) — `scripts/oracle-normalize.py`
  is a deliberate no-op scaffold, not an implementation.
- Growing the oracle corpus beyond the "handful" the acceptance criterion
  asks for, or covering non-scalar printed values.
- Any refactor of `test/integration/`'s existing lit config, `not.py`, or
  `nora-lit.in` — `test/oracle/` copies the working pattern rather than
  factoring out a shared lit-config helper; a shared-helper refactor is
  tempting but is exactly the kind of unrelated cleanup this stage's
  constraints say not to bundle in.
