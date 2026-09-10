# PLAN — issue #123: M2/GC S7, char / void / null / eof become immediates

## 1. Problem restated

S7 (docs/value-model-gc-migration.md §4, Phase 2) extends the S6 pattern
(`Value::immediate`, `nr_value` words, materialize-on-demand at legacy
boundaries) to the remaining Phase-2 leaf types. Auditing the current tree
narrows this considerably from what the title suggests:

- **`Char`** (`AST.h:417`) is a real, heap-allocated `SmallString<8>`-backed
  node, constructed at parse time for every character literal
  (`Parse::parseChar`, `Parse.cpp:449`) and compared/printed by echoing that
  stored string. It is the type the master doc names explicitly ("retires
  Char's `SmallString<8>`"). **But `Interpreter::visit(ast::Char const&)`
  (`Interpreter.cpp:776`) is unreachable today** — `parseExpr` (`Parse.cpp:211`)
  never calls `parseChar`, only `parseValue` (the *quoted-datum* parser,
  `Parse.cpp:324`) does, and a quoted char's containing `QuotedExpr` is cloned
  as one opaque unit by `visit(ast::QuotedExpr const&)` (`Interpreter.cpp:804`)
  without ever re-dispatching into the wrapped `Char`. So today a bare
  `#\a` doesn't parse at all, and no program can reach the visitor overload
  this slice needs to change. Real Racket treats characters as self-evaluating
  (exactly like booleans, integers, strings — verified: `(display #\a)` reads
  and evaluates directly), and `parseExpr` already treats
  `Integer`/`BooleanLiteral`/`String` this way (`Parse.cpp:218-233`) but never
  gained the equivalent line for `Char` (or `Vector`, which stays out of
  scope — see §6). This plan **bundles a four-line grammar fix** (add
  `parseChar` to `parseExpr`, mirroring the existing `parseBooleanLiteral`
  call) into the slice: without it, `Value::immediate`/`materializeLegacy`
  dispatch for chars would be untested dead code, and the issue's own forcing
  criterion ("guarded by char/void `.rkt` tests") is unsatisfiable without a
  way to write an unquoted char `.rkt` test in the first place.
- **`Void`** (`AST.h:933`) already has no payload; the only work is routing
  its two procedural producers — `step(Frame::Define&)`
  (`Interpreter.cpp:303`, `define-values`) and `step(Frame::Set&)`
  (`Interpreter.cpp:338`, `set!`) — through `Value::immediate(NR_VOID)`
  instead of `make_unique<ast::Void>()`, plus the dead-but-consistent
  `visit(ast::Void const&)` (`Interpreter.cpp:800`).
- **`Null`/`Eof` have no existing runtime representation to migrate.** There
  is no `ast::Null`/`ast::Eof` class, no `'()` singleton (the empty list is
  today just a zero-element `ast::List`, `AST.h` — a real container type,
  not a value this ladder's Phase 2 touches), and no `eof-object`/`null?`/
  `pair?`/`void` builtin exists in `Runtime.cpp`'s function table
  (`Runtime.cpp:490-511`). `NR_NULL`/`NR_EOF` are already defined in
  `nora_rt.h:42,44` (from S0) and need no change — they simply have no
  producer yet. Introducing `'()`-as-immediate, `eof-object`, `null?`, or
  `pair?` is tracked separately under M4 (#96, #147, #135 — "Unify List with
  cons Pairs so car/cdr/pair? span quoted lists"), not M2. **This plan
  migrates the two types that actually exist (`Char`, `Void`) and explicitly
  defers `Null`/`Eof`** — there is nothing to convert. The PR description
  must say this plainly so "closes #123" isn't read as silently dropping
  scope; S8 (#124, fixnums) depends only on the migrated set, not on
  Null/Eof.
- **A real, load-bearing side effect**: today `Char` stores the *original
  spelling* a character was written with, not its canonical codepoint —
  e.g. `'#\null` prints back as `#\null` (`test/integration/char-null-read.rkt`
  pins this), even though `'#\null` and `'#\nul` denote the same character.
  Verified against the installed `racket` (`/usr/bin/racket`): real Racket
  canonicalizes both to `#\nul` on `write`/`display`, and issue #73 (closed,
  deferred) already documents that `charReprFromCodePoint`'s names (`nul`,
  `vtab`, `page`, `rubout`, and the hex-escape fallback for other controls)
  are Racket's *correct* print names. Reducing `Char` to a bare codepoint
  (the immediate can only carry a codepoint — `nr_char`'s tag/shift encoding
  has no room for "which alias was typed") necessarily canonicalizes this,
  fixing a latent divergence from real Racket. `char-null-read.rkt`'s
  `CHECK:` line changes from `#\null` to `#\nul` **in the same commit**,
  with the rationale recorded in the commit message.

## 2. Files to touch

- `src/include/AST.h`:
  - `Char` (`AST.h:417-435`) — replace `llvm::SmallString<8> Value` with a
    plain `uint32_t CodePoint`; constructor becomes
    `explicit Char(uint32_t CodePoint)`; rename `getValue()` →
    `getCodePoint()`. Add two **inline, in-class-body** static members —
    they must be written directly inside the class body, not declared here
    and defined out-of-line in `AST.cpp` (§5 R0 explains why):
    - `static std::string reprFor(uint32_t CP)` — Racket's printed spelling
      for a codepoint (the body of today's file-local
      `charReprFromCodePoint`, `Parse.cpp:395`, moved here verbatim; still
      uses `std::snprintf` for the hex-escape fallback, so add `<cstdio>`).
    - `static std::optional<uint32_t> codePointForName(llvm::StringRef Name)`
      — the reverse of the 13-name table in `Lex.cpp:258-259`
      (`space`→0x20, `newline`→0x0A, `alarm`→0x07, `backspace`→0x08,
      `delete`→0x7F, `escape`→0x1B, `null`→0x00, `return`→0x0D, `tab`→0x09,
      `nul`→0x00, `vtab`→0x0B, `page`→0x0C, `rubout`→0x7F), returning
      `std::nullopt` for an unrecognized name.
  - Add `#include <cstdint>`, `#include <cstdio>`, `#include <string>`,
    `#include <llvm/ADT/StringRef.h>` (misc-include-cleaner: these are now
    used directly, verify at implementation time whether the last is already
    pulled in transitively by `llvm/ADT/SmallString.h`).
  - `Void` (`AST.h:933-950`) — unchanged.
- `src/AST.cpp`:
  - `Char::dump()`/`Char::write()` (`AST.cpp:118-125`) — call
    `Char::reprFor(CodePoint)` instead of echoing the stored string. These
    are pre-existing out-of-line member definitions, so growing their bodies
    is fine (§5 R0); do not extract a *new* out-of-line helper here.
- `src/Parse.cpp`:
  - Add `#include <llvm/Support/ConvertUTF.h>` directly (today only
    `UTF8.h`/`UTF8.cpp`/`Lex.cpp` include it; `Parse.cpp` needs its own
    direct include per misc-include-cleaner).
  - `Parse::parseChar` (`Parse.cpp:449`) — decode all three token kinds to a
    `uint32_t` directly, inline in this existing method body (no new
    function): `CHAR_NAMED` via `ast::Char::codePointForName(T.Value)` —
    if that returns `std::nullopt` (R4), handle it with a defensive
    diagnostic/early-return in the same shape this function already uses
    for other failures, **not** `assert(...)` (§5 R0: a new `assert` on a
    changed `Parse.cpp` line trips `misc-static-assert`); `CHAR` via UTF-8
    decode of `T.Value` with `llvm::convertUTF8Sequence`; `CHAR_HEX` via the
    existing hex-digit loop in `decodeHexChar` (`Parse.cpp:431`), inlined
    directly (it no longer needs to produce a display string, just the
    codepoint — so it stops calling `charReprFromCodePoint`).
  - **Delete** `charReprFromCodePoint` (`Parse.cpp:395`), `appendUTF8`
    (`Parse.cpp:374`), and `decodeHexChar` (`Parse.cpp:431`) as free
    functions — their logic moves to `Char::reprFor`/inlined `parseChar` per
    above. Do not leave them as dead code, and do not replace them with a
    *new* free function of the same shape (§5 R0).
  - `Parse::parseExpr` (`Parse.cpp:211`) — add a `parseChar` call
    immediately after the existing `parseBooleanLiteral` block
    (`Parse.cpp:223-226`), before `parseString`, mirroring that block
    exactly:
    ```cpp
    std::unique_ptr<ast::Char> C = parseChar(S);
    if (C) {
      return C;
    }
    ```
    `parseValue`'s existing `parseChar` call (`Parse.cpp:314`) is untouched.
- `src/include/Value.h`:
  - Add `static std::unique_ptr<ast::ValueNode> viewOf(nr_value W)` — a
    **public**, inline-in-class-body dispatcher shared by
    `materializeLegacy()` and `Interpreter::getResult()` (below), so the
    per-kind mapping lives in exactly one place (S8 will extend this same
    function for fixnums rather than re-deriving a second copy):
    ```cpp
    static std::unique_ptr<ast::ValueNode> viewOf(nr_value W) {
      if (nr_is_char(W)) {
        return std::make_unique<ast::Char>(nr_char_val(W));
      }
      if (W == NR_VOID) {
        return std::make_unique<ast::Void>();
      }
      // NOLINTNEXTLINE(misc-static-assert): a runtime check, not a constant.
      assert(W == NR_TRUE || W == NR_FALSE);
      return std::make_unique<ast::BooleanLiteral>(nr_truthy(W));
    }
    ```
  - `materializeLegacy()` (`Value.h:116-123`) — replace the
    boolean-only assert+construct with `Legacy = viewOf(Imm); Imm = 0;`.
- `src/include/Interpreter.h`:
  - `getResult()` (`Interpreter.h:85-94`) — replace the
    `nr_truthy`/`BooleanLiteral` special case with
    `return Value::viewOf(Result.rawImmediate());`.
- `src/Interpreter.cpp`:
  - `visit(ast::Char const &C)` (`Interpreter.cpp:776`) — deliver
    `Value::immediate(nr_char(C.getCodePoint()))` instead of cloning (mirrors
    `visit(ast::BooleanLiteral const&)`, `Interpreter.cpp:764`).
  - `visit(ast::Void const &Vd)` (`Interpreter.cpp:800`) — deliver
    `Value::immediate(NR_VOID)` instead of cloning (kept for consistency
    even though unreachable — no interpreter path should allocate a Void).
  - `step(Frame::Define &K)` (`Interpreter.cpp:303`, both `deliver` calls at
    lines 311 and 335) and `step(Frame::Set &K)` (`Interpreter.cpp:338`,
    `deliver` at line 349) — `deliver(Value::immediate(NR_VOID))` instead of
    `deliver(std::make_unique<ast::Void>())`.
- `src/ASTRuntime.cpp`:
  - `valueEq` (`ASTRuntime.cpp:102`), the `AST_Char` case (`ASTRuntime.cpp:123`)
    — compare `llvm::cast<Char>(A).getCodePoint() == ...getCodePoint()`
    (a `uint32_t ==`, replacing the `StringRef ==`). This incidentally fixes
    `(eq? '#\nul '#\null)` from `#f` to the Racket-correct `#t` — same root
    cause as the `write()` canonicalization in §1, worth one boundary test
    (§3 slice 4).
- `test/unit/test_interpreter.cpp`:
  - Extend `Run` (`test_interpreter.cpp:24-58`) with
    `expectChar(const ast::ValueNode*, uint32_t)` /
    `expectChar(uint32_t) const` and `expectVoid(const ast::ValueNode*)` /
    `expectVoid() const`, mirroring `expectBool` exactly.
  - New `TEST_CASE`s per §3.
- `test/integration/` — new files per §3; one existing file edited
  (`char-null-read.rkt`).
- No `src/mlir/`, `src/include/nir/`, `expander/expander.rktl`, or `docs/`
  changes.

## 3. TDD slices

Each slice is its own commit (squash merges are disabled on this repo).

1. **RED → GREEN: `Char` retires its string buffer; output is canonicalized.**
   Test: update `test/integration/char-null-read.rkt`'s `CHECK:` from
   `#\null` to `#\nul` **first** (RED against today's code, which still
   echoes `null`) — add a one-line comment recording that this now matches
   real Racket (`racket -e "(write '#\null)"` → `#\nul`) and issue #73.
   Keep every other existing `char-*.rkt`/`quote1{4,7,8}.rkt` test
   unchanged as a characterization net (`page`/`rubout`/`vtab`/`space`/
   `newline`/glyph names are all already Racket-canonical, so their
   `CHECK:` lines don't move).
   Production code: `Char` → `uint32_t CodePoint` (§2), `reprFor`/
   `codePointForName` added inline in `AST.h`, `Char::write`/`dump`
   (`AST.cpp`) call `reprFor`, `Parse::parseChar` decodes all three token
   kinds to a codepoint inline, `charReprFromCodePoint`/`appendUTF8`/
   `decodeHexChar` deleted from `Parse.cpp`, `ASTRuntime.cpp`'s `valueEq`
   compares codepoints. No behavior change is visible yet at the
   `Value`/immediate layer — this slice is purely "retire the buffer, keep
   equivalent output except the one intentional fix."
   Gate: run `debug`, `asan`, `ubsan` presets here — `SmallString<8>` had a
   possible heap spill (e.g. `"backspace"` is 9 bytes, over the inline
   capacity) that a trivially-destructible `uint32_t` cannot leak or
   double-free; confirming this under sanitizers before the immediate-wiring
   slice isolates any regression to the representation change, not the
   `Value` plumbing.

2. **RED → GREEN: chars become self-evaluating and deliver as immediates.**
   Test: new `TEST_CASE("bare char literal result is the nr_char immediate,
   not an allocated Char", "[interp][m2][gc]")` — `runLinklet("(linklet () ()
   #\\a)")`, assert `R.RawImmediate == nr_char('a')` and
   `R.expectChar('a')`. RED because `parseExpr` doesn't parse a bare `#\a`
   yet (fails at the `REQUIRE(AST)` in `runLinklet`), and
   `Value::immediate`/`viewOf` don't handle chars yet.
   Also add: `test/integration/char-self-eval.rkt` (`(linklet () () #\a)` →
   `CHECK: #\a`), `test/integration/char-self-eval-glyph.rkt` (`#\λ` →
   `CHECK: #\λ`), `test/integration/char-self-eval-named.rkt` (`#\space` →
   `CHECK: #\space`) — the first unquoted-char coverage in the repo; quoted
   char tests (`char-glyph-read.rkt` et al.) exercise a structurally
   different path (`visit(QuotedExpr)` clone, §5 risk R2) and don't touch
   this.
   Production code: `parseExpr` gains the `parseChar` call (§2),
   `visit(ast::Char const&)` delivers `Value::immediate(nr_char(...))`,
   `Value::viewOf`/`materializeLegacy` dispatch on `nr_is_char`,
   `Interpreter::getResult()` routes through `viewOf`.

3. **RED → GREEN: `Void` becomes an immediate at its two producers.**
   Test: new `TEST_CASE("define-values result is the NR_VOID immediate,
   not an allocated Void", ...)` — `runLinklet("(linklet () ()
   (define-values (x) 5))")`, assert `R.RawImmediate == NR_VOID` and
   `R.expectVoid()`. A second case for `set!`:
   `runLinklet("(linklet () () (let-values ([(x) 1]) (set! x 2)))")`,
   asserting `R.expectVoid()`. If `R.RawImmediate` doesn't also come back
   `NR_VOID` for this second case, that's the already-documented Call/
   WcmMark materialization deferral (§5 R8), not a bug to chase here — keep
   `expectVoid()` as the assertion for it either way.
   No new integration test: `Void::write()` is a no-op
   (`AST.cpp:568`, unchanged), so a linklet whose result is void prints an
   empty line — asserting that via `FileCheck` (`CHECK-EMPTY:` +
   `--strict-whitespace --match-full-lines`) is a weak, whitespace-fragile
   signal compared to the direct `RawImmediate == NR_VOID` unit assertion,
   and every existing `.rkt` test already has a non-void last form, so
   there's no regression risk to characterize either. Skipped deliberately,
   not an oversight.
   Production code: `step(Frame::Define&)`/`step(Frame::Set&)`/
   `visit(ast::Void const&)` deliver `Value::immediate(NR_VOID)`;
   `Value::viewOf` already dispatches `NR_VOID` from slice 2's `viewOf`
   addition (this slice just adds the producer side).

4. **RED → GREEN: legacy call boundaries and `eq?` for the newly-immediate
   types.**
   Tests (unit, mirroring S6 slice 3): `(eq? #\a #\a)` → `#t`,
   `(eq? #\a #\b)` → `#f` (no existing test passes a bare `Char` to a
   `RuntimeFunction`; `EqFunction`/`BoxFunction` dereference `Args[i]`
   unconditionally, so this would segfault today if `Value::get()` returned
   `nullptr` for an engaged char immediate), `(box #\a)` /
   `(unbox (box #\a))` → `#\a` as a second boundary (container
   construction). Also `(eq? '#\nul '#\null)` → `#t` (§2, the
   `valueEq` codepoint fix — this is quoted, so it exercises
   `ast::valueEq` directly, not the immediate path, but it's the same root
   cause and was previously untestable-as-a-regression since nothing pinned
   the old, wrong `#f`).
   New `test/integration/error-noncallable-char.rkt`: `(#\a 1 2)` →
   `CHECK: error: application: expected a procedure in operator position`
   (mirrors `error-noncallable-boolean.rkt`; `Value::operator bool()` already
   folds `Imm != 0` generically — this is a regression-pin, not new
   production code, since S6 already made `operator bool()` immediate-kind-
   agnostic).
   Production code: none expected (this slice should be pure test
   coverage); if `Value::get()`/`takeLegacy()`/`toShared()` don't already
   materialize a char/void immediate correctly, that's a bug introduced in
   slice 2/3 to fix here, not new scope.

5. **Final slice: sanitizer confirmation (a gate, no new test).**
   Run `ctest --preset asan` and `ctest --preset ubsan` in addition to
   `debug`, covering the full range of `Value`/`Char` ownership changes
   across slices 1-4 (per docs/value-model-gc-migration.md §6 R8's general
   caution about half-migrated ownership changes).

## 4. AST/visitor surface

No new `ASTNodeKind` and no visitor signature changes. `Char`
(`AST_Char`) and `Void` (`AST_Void`) keep their existing enum positions;
every `ASTVisitor`/`Interpreter`/`AnalysisFreeVars` `visit(...)` overload
list is untouched in shape — only bodies change (`visit(ast::Char const&)`,
`visit(ast::Void const&)` in `Interpreter.cpp`). `AnalysisFreeVars::visit`
for both (`AnalysisFreeVars.cpp:34,155`) are pure no-ops ("no free
variables") and are unaffected by the representation change.

## 5. Risk areas

- **R0 — the clang-tidy anon-namespace/free-function trap (the reason §2
  keeps insisting on "inline, in-class-body").** `.clang-tidy` in this repo
  enables **both** `misc-use-anonymous-namespace` (rejects a new `static`
  free function) **and** `llvm-prefer-static-over-anonymous-namespace`
  (rejects a new anonymous-namespace one) — they contradict each other on
  any *new* file-scope free function, and `.github/workflows/clang-tidy.yml`
  is diff-scoped (`clang-tidy-diff.py -warnings-as-errors`), so either shape
  fails CI on a changed line. The only escape is
  `AllowMemberFunctionsInClass`: a member function **defined inline inside
  the class body** is exempt; the same function declared in the header and
  **defined out-of-line** in a `.cpp` is not. Concretely: do **not** add
  `charReprFromCodePoint`/`decodeHexChar`/`appendUTF8`-shaped free functions
  back to `Parse.cpp`, and do **not** give `ast::Char::reprFor`/
  `codePointForName` out-of-line definitions in `AST.cpp` — write their
  bodies directly inside the `Char` class in `AST.h`. Growing the body of an
  *already-existing* out-of-line member (`Char::write`/`dump` in `AST.cpp`,
  `Parse::parseChar` in `Parse.cpp`) is fine — that's not a new function.
  Separately, `misc-static-assert` fires on `assert(cond && "msg")` on any
  changed **`.cpp`** line (headers aren't re-scanned this way, so the
  existing `Value.h` asserts are unaffected) — if `Char::codePointForName`
  can return `std::nullopt` for a name `Lex.cpp` already validated (R4),
  handle it in `Parse::parseChar` with a defensive early return
  (`Diag.error(...)` + `S.rewindTo(Start)`, matching this function's
  existing failure shape) — **not** `assert(CP.has_value() && "...")` —
  since that assert would sit on a changed `Parse.cpp` line and trip CI.
- **R1 — `write()`'s Racket-compatible output.** The one *intended* change
  (`#\null` → `#\nul`, §1) is pinned by slice 1's updated
  `char-null-read.rkt`. An *unintended*, currently-untested change: `alarm`
  (codepoint 7, BEL) and `escape` (codepoint 27, ESC) are accepted as
  named-char spellings by `Lex.cpp`'s over-permissive list (`Lex.cpp:258`)
  but have no case in `charReprFromCodePoint`/`reprFor` (verified against
  the installed `racket`: `(write (integer->char 7))` and
  `(write (integer->char 27))` both print as a 4-hex-digit escape form,
  not a name — real Racket doesn't accept `#\alarm`/`#\escape` as *read*
  syntax at all, so there's no canonical print name to preserve);
  `delete` (codepoint 127) is a valid spelling but canonicalizes to
  `#\rubout`. After this slice, typing `#\alarm`/`#\delete`/`#\escape`
  prints the hex-escape form / `#\rubout` / the hex-escape form instead of
  echoing the typed name. No existing test exercises any of these three
  names (confirmed via search), so nothing breaks, but it's a real, silent
  behavior change worth flagging in the PR description. Fixing the lexer's
  over-acceptance of names real Racket rejects is a separate, pre-existing
  issue (out of scope, §6).
- **R2 — quoted chars still bypass the immediate path entirely.** Exactly
  like S6's R4 for booleans: `visit(ast::QuotedExpr const&)`
  (`Interpreter.cpp:804`) clones the whole node, so `'#\a` never reaches
  `visit(ast::Char const&)` and stays a heap `Char` (now POD, but still
  legacy-materialized, never an `nr_value` immediate). All the existing
  `char-*.rkt`/`quote*.rkt` tests are quoted and therefore validate slice
  1's representation change but **not** slice 2's immediate-delivery path —
  that's why slice 2 adds dedicated unquoted `.rkt` tests.
  `Vector` has the identical grammar gap (`parseExpr` never calls
  `parseVector`) and is deliberately **not** given the same treatment here
  (§6) — this plan only closes the gap for `Char`, which this issue names.
- **R3 — the two-dispatch-copies trap.** Before this slice,
  `materializeLegacy()` and `getResult()` each independently special-cased
  boolean immediates; naively adding char/void cases to both would
  duplicate the mapping and let them drift when S8 adds fixnums. Guard:
  `Value::viewOf` (§2) is the single dispatcher both call.
- **R4 — the named-char table must track `Lex.cpp`'s list, not
  reinvent it.** `Char::codePointForName` (§2) must accept exactly the 13
  names `Lex.cpp:258-259` lexes (`space, newline, alarm, backspace, delete,
  escape, null, return, tab, nul, vtab, page, rubout`) with the same
  codepoints `charReprFromCodePoint`'s forward direction already assigns
  where they overlap (`nul`≡`null`→0, `tab`→9, `newline`→10, `vtab`→11,
  `page`→12, `return`→13, `space`→32, `rubout`≡`delete`→127,
  `backspace`→8) plus the two names that direction has no case for
  (`alarm`→7, `escape`→27 — R1). A mismatch here would make
  `Parse::parseChar` reject (or misdecode) a name the lexer already
  accepted — handle the gap defensively, not via `assert` (R0), and cross-
  check the table against `Lex.cpp:258-259` by hand at implementation time,
  not just against this plan's transcription of it.
- **R5 — free-variable analysis.** Zero risk: `AnalysisFreeVars::visit`
  for `Char`/`Void` are unconditional no-ops (§4) and don't inspect the
  node's payload.
- **R6 — memory safety.** `SmallString<8>` → `uint32_t` strictly reduces
  risk (a possible heap spill for names ≥9 bytes becomes impossible); the
  `Value`/`materializeLegacy`/`viewOf` changes are the same shape of
  ownership change S6 already made safe. Run `debug`/`asan`/`ubsan` at
  slice 1 (representation) and again at slice 5 (full gate) per docs/
  value-model-gc-migration.md §6 R8.
- **R7 — `RuntimeFunction`-produced `Void` stays heap-allocated (deferred,
  matches S6 precedent).** `SetBoxFunction`/`SetCarFunction`/`SetCdrFunction`
  (`Runtime.cpp:257,390,412`) still `return std::make_unique<ast::Void>()`
  through the untouched legacy `RuntimeFunction` ABI (`const ValueNode*` in,
  `unique_ptr<ValueNode>` out) — converting that boundary to immediates is
  the same out-of-scope ABI change S6's plan deferred for booleans (`eq?`/
  `zero?`), not something this slice's forcing tests require.
- **R8 — slice 3's `set!` case may not force through `RawImmediate`.** If
  the `let-values` body frame round-trips the `set!` result through
  `takeLegacy()` before it reaches `Result` (the same S6-deferred Call/
  WcmMark materialization pattern, §6), `R.RawImmediate` could legitimately
  come back `nullopt` even though the value is void. That's the known,
  already-documented deferral, not a slice-3 bug — if it happens, keep
  `R.expectVoid()` as the assertion for that case and don't spend time
  forcing `RawImmediate` through it. The top-level `define-values` case is
  the real forcing test and is unaffected (S6 already threads the last
  top-level form's result as a `Value` straight into `Result`, per
  `visit(ast::Linklet const&)`).

## 6. Out of scope

- **`Null`/`Eof` as producible values** (`'()` as an immediate,
  `eof-object`, `null?`, `pair?`, `void` as a callable). No current code
  path constructs either; introducing them is real feature work belonging
  to M4 (#96, #147, #135), not a Phase-2 "retire an existing
  representation" slice. `NR_NULL`/`NR_EOF` already exist in `nora_rt.h` and
  need no change.
- **`Vector` self-evaluation.** Has the identical `parseExpr` gap as `Char`
  had (R2), but the issue doesn't name it and it isn't a Phase-2 immediate
  candidate (vectors are containers, not leaves) — leave `parseExpr`
  untouched for `Vector`.
- **Fixing `Lex.cpp`'s over-acceptance of `alarm`/`delete`/`escape`** to
  match real Racket's stricter read grammar (R1). Pre-existing, unrelated
  to the value-representation migration.
- **Extending immediate-preservation through `Call`/`WcmMark` frames**
  (`step(Frame::Call&)`/`step(Frame::WcmMark&)` round-trip through
  `takeLegacy()`, materializing on the way out of a lambda call or
  `with-continuation-mark`). Identical, already-documented S6 deferral —
  not required by this issue's forcing tests, and bundling it conflates
  "make these literals immediate" with "make call/mark plumbing
  allocation-free."
- **Converting `RuntimeFunction`-produced `Void` to an immediate** at the
  `applyProcedure` `deliver(std::move(R))` boundary (R7) — same ABI-boundary
  deferral S6 made for `eq?`/`zero?`-produced booleans.
- **S8 (fixnums → `nr_fixnum`).** Next slice in the ladder (#124); this plan
  does not touch `Integer`.
- Any `docs/`, MLIR (`src/mlir/`, `src/include/nir/`), or
  `expander/expander.rktl` changes.
- Formatting/lint-only cleanup beyond what `clang-format`/`clang-tidy`
  require for the new/changed lines.
