# NORA Roadmap — From FEP Interpreter to a Compiled `#lang racket/base` Hello-World

## Current state

`norac` parses **one** self-contained linklet (a Fully-Expanded-Program subset) and interprets it with a CEK/CESK machine (explicit typed `Frame` stack in a heap `std::vector<Kont>`). The constraints that dominate everything below, verified against the tree:

- **No proper tail calls.** `applyProcedure` unconditionally `Kont.emplace_back(Frame::Call)` (`src/Interpreter.cpp:510`); a `Call` frame is popped only when its activation returns a value (`:375`). Tail loops grow `Kont` by one frame per iteration — O(depth) heap where Racket guarantees O(1). The expander is deeply/mutually tail-recursive (`for-loop` 563×, `loop` 172×), so this is disqualifying. The machine is *already* a trampoline, which makes the fix a local one (reuse the `Call` frame) rather than an architecture change.
- **Value-copy model, no shared heap, no GC, no identity.** Values are `std::unique_ptr<ast::ValueNode>` deep-**cloned** on every bind/lookup (`Environment::envLookup` returns `clone()`). `ast::List` is a flat `SmallVector`, not cons cells (so `cons` is O(n), list-building O(n²)). Symbols are not interned (`AST.h:212` FIXME); `eq?`/`eqv?`/`equal?` are stubbed by a name-comparing `valueEq`. All integers are GMP `mpz_class` (RAII) — no fixnums, flonums, rationals, or complex. Cycle teardown is a manual `AllScopes` hack, not a collector.
- **6 primitives total** (`+ - *`, `current-continuation-marks`, `continuation-mark-set-first`, `continuation-mark-set->list`). Verified: no `box`/`cons`/`eq?`/`car`/`cdr` exist anywhere in `src/`. The expander needs **544 distinct kernel primitives** (measured by the R3 spike — `tools/freevars.py` → [`docs/expander-primitive-surface.md`](docs/expander-primitive-surface.md)); *running* a `racket/base` program needs a second, larger wave beyond that (see M16).
- **Linklets are parsed but not linked.** `visit(Linklet)` builds one `GlobalEnv` from body `define-values` and keeps the last form's value; imports are rejected by the parser, exports are unused. No instance objects, no import/export wiring, no `#%linklet` runtime API. Free identifiers resolve **lazily at eval time**, not eagerly at link time.
- **NIR is an empty, partly-broken opt-in scaffold** (`src/include/nir/`, `src/mlir/`): one `UnimplementedOp`, a stale duplicate `Ops.td` with the obsolete `list<OpTrait>` API, class-name mismatches, never linked into the interpreter. Rewrite, don't extend.
- ~88–90 integration `.rkt` tests (lit/FileCheck) + 1 Catch2 unit file. **No Racket oracle infrastructure exists yet.**

The expander artifact (`expander/expander.rktl`, 8.6 MB) is a single `(linklet () (<71 exports>) <body>)` with **empty imports** — every kernel primitive is a **free variable** the host must supply — **~2875 internal `define-values`** (~2000 at top level), and **71 exports** (`boot`, `eval`, `expand`, `compile`, `read`, `datum->syntax`, `make-namespace`, `namespace-require`, `dynamic-require`, …). Its body tail *is* the boot sequence. It also consumes the linklet runtime itself as free-variable primitives (`compile-linklet`, `instantiate-linklet`, `make-instance`, `primitive-table`, …) — the expander is a linklet that compiles and runs linklets by calling back into the host.

## Target

A **standalone AOT executable** `./hello`, produced from `hello.rkt` (`#lang racket/base`), that prints `hello world`:

```
hello.rkt (#lang racket/base)
  --[NORA's reader + #lang/module-reader protocol]-->     (real read-syntax over a file port)
  --[NORA's own interpreter runs expander.rktl]-->        (self-hosted expander, phase-1 macros execute)
expanded FEP / linklet-bundle graph (+ racket/base graph deps)
  --[NORA backend: FEP -> NIR (MLIR) -> LLVM IR]-->        (Racket-level opts in NIR)
object files + libnora_rt (Boehm GC + primitives + printer + main())
  --[link]--> ./hello  ->  prints "hello world"
```

## Issue tracking

Every milestone and spike below is a GitHub issue (label `roadmap`), cross-linked by dependency. Browse: <https://github.com/pmatos/nora/issues?q=is%3Aissue+label%3Aroadmap>

- **Spikes:** [R1 #88](https://github.com/pmatos/nora/issues/88) · [R3 #89](https://github.com/pmatos/nora/issues/89) · [R5 #90](https://github.com/pmatos/nora/issues/90)
- **Track A (interpreter → self-hosted expander):** [M0 #91](https://github.com/pmatos/nora/issues/91) · [M0-N #92](https://github.com/pmatos/nora/issues/92) · [M1 #93](https://github.com/pmatos/nora/issues/93) · [M2 #94](https://github.com/pmatos/nora/issues/94) · [M3 #95](https://github.com/pmatos/nora/issues/95) · [M4 #96](https://github.com/pmatos/nora/issues/96) · [M5 #97](https://github.com/pmatos/nora/issues/97) · [M6 #98](https://github.com/pmatos/nora/issues/98) · [M7 #99](https://github.com/pmatos/nora/issues/99) · [M8 #100](https://github.com/pmatos/nora/issues/100) · [M9 #101](https://github.com/pmatos/nora/issues/101) · [M10 #102](https://github.com/pmatos/nora/issues/102) · [M11 #103](https://github.com/pmatos/nora/issues/103) · [M12 #104](https://github.com/pmatos/nora/issues/104) · [M13 #105](https://github.com/pmatos/nora/issues/105) · [M14 #106](https://github.com/pmatos/nora/issues/106) · [M15 #107](https://github.com/pmatos/nora/issues/107) · [M16 #108](https://github.com/pmatos/nora/issues/108) · [M17 #109](https://github.com/pmatos/nora/issues/109)
- **Track B (NIR/LLVM AOT backend):** [B0 #110](https://github.com/pmatos/nora/issues/110) · [B1 #111](https://github.com/pmatos/nora/issues/111) · [B2 #112](https://github.com/pmatos/nora/issues/112) · [B3 #113](https://github.com/pmatos/nora/issues/113) · [B4 #114](https://github.com/pmatos/nora/issues/114) · [B5 #115](https://github.com/pmatos/nora/issues/115)

## Toolchain version lock (governs everything below)

All version-sensitive artifacts **must** derive from **one pinned Racket commit**, recorded as a single controlled fact (`ORACLE_RACKET_COMMIT`):

- `expander/expander.rktl` (regenerated via the `raco` demodularizer),
- the differential oracle (M0),
- the precompiled/embedded `racket/base` `.zo` graph (M16),
- the fasl v2/v3 wire format the deserializer targets (M13),
- the `primitive-table` name→module grouping the expander expects (M8).

A mismatch in any one silently invalidates **every** M10+ differential test. Regenerating `expander.rktl` + the `racket/base` `.zo` graph from that pinned build is a **scheduled build-infra task** (M0 deliverable) — the `raco` demodularizer toolchain is part of the budget, not free.

## Strategy — two interleaved tracks over one shared substrate

Build **one shared prerequisite substrate** first — the *spine* — then run two tracks that share a runtime and converge at the end:

- **Spine (critical path, blocks both tracks):** a single **value model + garbage-collected heap + object identity + proper tail calls + closure representation**, whose representation ABI is **co-designed once** and used by both the interpreter and compiled code. This is the pivot; getting it wrong forces rewriting both tracks.
- **Track A — interpreter → self-hosted expander:** grow the primitive surface (6 → 544), add the literal/value types the expander's data needs, build the printer, build linklet instances + the `#%linklet`/`primitive-table` runtime API, instantiate `expander.rktl`, prove phase-1 macro execution, add the fasl/`.zo` loader, the reader/`#lang` protocol, and the `racket/base` graph, culminating in an *interpreted* `#lang racket/base` hello. This is the long pole; staff it heaviest.
- **Track B — NIR/LLVM AOT backend:** rewrite the NIR dialect for real, prove the whole toolchain on a trivial FEP→NIR→LLVM→`./out` walking skeleton (needs **no** GC, can start day one), then build the runtime lib on the frozen M2 ABI **sharing Track A's primitive library**, cover all FEP forms, and link a linklet-bundle graph into an executable.

**Honest parallelism.** Only **B0 and B1 run genuinely ahead** of Track A — a trivial `(+ 1 2)`/`if` → `./out prints 42` needs no GC, no identity, no primitives. Everything past B1 interleaves with Track A: **B2 shares the M4–M7 primitive library and printer, B4 depends on M8's instance model.** Racket-as-oracle (differential testing, never as the expander) still decouples *test authoring* — B2–B4 can be exercised on Racket-produced `racket/base` FEP fixtures — but their *acceptance* ("compile and run a `racket/base` fixture") is gated on the shared primitives existing. The claim "Track B races far ahead in parallel" is false and is not planned around.

Legend: **[S]** spine · **[A]** Track A · **[B]** Track B. Effort: S=small, M=medium, L=large, XL=extra-large. Coarse person-time in each milestone header is *individual* duration, not additive across the serial chain — see "Effort realism".

---

## Milestone ladder

### M0 — Oracle + differential harness + version lock **[S] · ~2–3 wk**  ·  [#91](https://github.com/pmatos/nora/issues/91)
- **Goal:** Make Racket usable as a differential oracle; lock behavior and provenance.
- **Deliverables:** `test/oracle/` harness with a `%racket` lit substitution that runs an expression through the **pinned** `racket` and `norac`, normalizes printed output, and diffs; gated on `NORA_HAVE_RACKET` (auto-skips green when absent). Record `ORACLE_RACKET_COMMIT`; a build-infra task that **regenerates `expander.rktl` (via the `raco` demodularizer) and the `racket/base` `.zo` graph from that commit**. New opt-in CI `oracle` job; new CMake `lsan` preset. **The gensym/alpha-normalizer is scaffolded here but scoped to M-effort and lands for real just before M10** (see M0-N).
- **Depends-on:** —
- **Acceptance:** `ctest` runs a handful of `.rkt` files through both engines and passes; harness skips green with no Racket; `expander.rktl` and the `.zo` graph both carry a recorded provenance stamp matching `ORACLE_RACKET_COMMIT`.
- **Risk:** Unpinned oracle → flaky diffs; a provenance mismatch invalidates M10+.

### M0-N — Global-consistent alpha/gensym normalizer **[S→M] · ~3–4 wk**  ·  [#92](https://github.com/pmatos/nora/issues/92)
- **Goal:** A canonicalizer that can diff fully-expanded programs. Not "small": it must handle lifts, `let-values` temporaries, module-path-index encodings, and scope-annotated identifiers.
- **Deliverables:** a normalizer that renames **globally and consistently** across the whole expanded program (not local alpha-renaming), tolerant of a shifted gensym counter (every generated name renumbered coherently). A fallback "observational-equivalence" acceptance mode for cases text-normalization cannot reconcile.
- **Depends-on:** M0.
- **Acceptance:** normalizes non-trivial *real expander output* (not toy cases) to a stable canonical form; two runs of the same input over the pinned Racket normalize identically.
- **Risk:** See R7 — hash-iteration order and gensym-counter parity can defeat any normalizer; this milestone gates M10 and must be proven on real output before M10 is declared reachable.

### M1 — Proper tail calls in the interpreter **[S] · ~3–4 wk**  ·  [#93](https://github.com/pmatos/nora/issues/93)
- **Goal:** O(1)-space tail calls in the CEK machine.
- **Deliverables:** static tail-position marking (a `bool Tail` on `ExprNode`/visit); `Seq`/body/`let`/`letrec`/`let-values` frames popped **before** their last sub-expression (mirroring the existing `IfBranch` at `:165`); `applyProcedure` **reuses** the enclosing `Call` frame in tail position (transfer callee ownership + replace marks) instead of `emplace_back` at `:510`.
- **Depends-on:** M0.
- **Acceptance:** `(let loop ([n 10000000]) (if (= n 0) 'ok (loop (- n 1))))` → `ok` with **bounded** peak `Kont` (asserted via a test hook exposing peak `Kont` size); asan/ubsan clean.
- **Risk:** Fragile "peek the top frame" heuristics — use static tail annotation, the standard CEK approach. Independent of GC; runs alongside M2 prep.

### M2 — Shared value model + GC + closure representation (the pivot) **[XL] · ~3–5 mo**  ·  [#94](https://github.com/pmatos/nora/issues/94)
- **Goal:** Replace clone/value-copy with a shared, identity-preserving, GC-managed heap, and fix closure capture. **This is the single ABI decision point for both tracks — freeze it here.**
- **Deliverables:**
  - a tagged 64-bit `nr_value` word (low-bit/3-bit tagging, Chez/CS-style: fixnum immediate, char/`#f`/`#t`/`null`/`void`/`eof` immediates, 8-byte-aligned heap pointers); a uniform `ObjHeader` (type tag + GC bits + len/meta).
  - **Boehm–Demers–Weiser conservative GC (`libgc`) for both interpreter and compiled code — committed, not "first."** A conservative collector scans the native stack, which is exactly what the compiled `tailcc` path needs. **Moving/precise GC (stack maps/statepoints/shadow stack) is explicitly out of scope** for hello and filed as a future XL that would itself change the ABI; there is **no** "reserved forwarding slot" pretense.
  - GMP routed through GC: an explicit task to **migrate off `mpz_class` (RAII) to GC-routed raw `mpz_t`**, using Boehm atomic allocation for pointer-free limbs, so bignum `nr_value`s need no finalizers. Own test.
  - `Environment` returns **shared** references, not clones. **Closure representation designed here: a flat closure = header + code-ptr + vector of captured cells**, aligned byte-for-byte with B2's compiled closure layout, so a lambda capturing over the ~2000-entry top level is O(captured), not O(env). No naive env chaining/copying.
  - a minimal `cons`/`eq?`/`box`/`unbox`/`set-box!` quintet pulled in here (they are the natural first identity/mutation smoke tests). `eq?`/`eqv?`/`equal?` cycle-safe (seen-set). Interned symbols. `clone()` and the `AllScopes` teardown deleted.
- **Depends-on:** M1; spike **R1** first.
- **Acceptance:** a **C++-level `nr_value` unit test** proves pointer identity and mutation without surface syntax; `(let ([p (cons 1 2)]) (eq? p p))` → `#t`; `(let ([b (box 0)]) (set-box! b 5) (unbox b))` → `5`; a tail loop allocating ≫ heap-size of garbage completes (GC demonstrably collects, RSS bounded); all ~90 tests + asan/lsan green.
- **Risk:** Broad refactor of `AST.h`/`ASTRuntime.h`/`Environment`/`Runtime.cpp`. Migrate one type at a time behind the `ValueNode` interface with golden tests green at each step. **Do this before growing the primitive table** — every primitive signature depends on it.

### M3 — Literal reader coverage + literal value types + bounded parsing **[L] · ~1–1.5 mo**  ·  [#95](https://github.com/pmatos/nora/issues/95)
- **Goal:** Parse the whole 8.6 MB expander artifact and construct its data literals.
- **Deliverables:** lexer/parser for flonums, exact rationals (`1/2`), byte strings (`#"…"`), `#hash(...)`/`#hasheq(...)` reader syntax, and acceptance of `define-syntaxes` (58×, must parse in this flattened artifact); value types `Flonum` (double, boxed), `Rational` (GMP `mpq`), `ByteString`, immutable `Hash`, plus fixnum/bignum promotion on `nr_value` (fixnum fast-path via `__builtin_*_overflow`, mpz fallback). **Bounded/iterative parsing (or a raised parser stack) for deeply-nested data**, since the recursive-descent parser will otherwise blow the C++ stack on this file.
- **Depends-on:** M2.
- **Acceptance:** `norac` parses `expander.rktl` end-to-end without a lex/parse error **and without stack overflow** (parse-only smoke test on the real artifact + a pathological-depth fixture); differential tests for each literal kind.
- **Risk:** Today `0.0`/`1/2`/`7/8` lex as **identifiers** — the file will not even parse until this lands; deep nesting is an independent failure mode.

### M4 — Core datatypes, equality & the numeric tower **[L] · ~1.5–2 mo**  ·  [#96](https://github.com/pmatos/nora/issues/96)
- **Goal:** The pure-value layer the expander builds at load time.
- **Deliverables:** pairs/lists (`car cdr cons null? pair? list length list* append apply map …`), mutable pairs, mutable/immutable strings (real code-unit buffers — drop the "lexeme-includes-quotes" hack) and chars (immediate codepoint), bytes, vectors (+ `unsafe-vector*-ref/-set!/-length`), boxes, `void`/`eof`; `values`/`call-with-values`; interned symbols + gensym; fixnum/`unsafe-fx*` ops + generic-int subset (`add1 sub1 zero? exact-integer? fixnum? number? quotient expt`). Completes the **GMP raw-`mpz_t`** migration for the bignum arithmetic paths.
- **Depends-on:** M2 (M3 in parallel).
- **Acceptance:** each of the top-50 datatype/numeric prims passes a `norac`-vs-`racket` differential test.
- **Risk:** `unsafe-*` ops are a bit-layout contract — they must assume the tag with no dispatch; co-designed with M2.

### M5 — Structs, properties, hash tables (iteration-order-matched) **[L] · ~2–3 mo**  ·  [#97](https://github.com/pmatos/nora/issues/97)
- **Goal:** The heaviest expander object surface — everything in the expander is a struct or a hash.
- **Deliverables:** struct system (`make-struct-type` 156–190×, `make-struct-field-accessor` **506×**, `make-struct-field-mutator`, `make-struct-type-property`, `struct-copy` 136×, `struct->vector`, `current-inspector`) with property values (`prop:authentic` 92×, `prop:equal+hash`, `prop:procedure`, `prop:custom-write`, …); prefab structs (global prefab table); `eq`/`eqv`/`equal` hashes — mutable (`hash-set!` 119×) and immutable HAMT (`hash-set` 167×, structural sharing) — plus the `hash-iterate-*` protocol (incl. `unsafe-immutable-hash-iterate-*`) and `#hash`/`#hasheq` literals.
- **Depends-on:** M4.
- **Acceptance:** differential tests for struct construction/accessor/mutator/property/`struct-copy` and hash CRUD+iteration. **Additionally: the `eq`/`equal`-hash iteration order either matches Racket CS or is proven unused for code-emission ordering** (see R7) — this is a gating acceptance criterion, not a footnote.
- **Risk:** Load-bearing #1 after pairs; a subtly wrong `make-struct-type`/`hash-ref` corrupts the expander invisibly. Hash iteration order silently threatens *every* downstream expand-equivalence test.

### M6 — Printer + errors, exceptions, parameters & control **[L] · ~1.5–2 mo**  ·  [#98](https://github.com/pmatos/nora/issues/98)
- **Goal:** The value printer plus the control/diagnostic layer the expander touches on any non-trivial path. **The printer moves here (before/with the error system), because error text is formatted through it.**
- **Deliverables:** a real `write`/`display`/`print` subsystem (cycle handling, `prop:custom-write`) — the offending-value formatter that `raise-argument-error` and friends depend on; `raise` / `error` (115×) / `raise-argument-error` (**489×**) / `raise-arguments-error`, the `exn`/`exn:fail` struct hierarchy, `with-handlers`→`call-with-exception-handler`; parameters built on continuation marks (`make-parameter`, `parameterization-key`, `extend-parameterization`, `parameterization?`); `dynamic-wind`, `call-with-continuation-prompt`, `abort-current-continuation` for the eval loop (the marks 3-op family already exists).
- **Depends-on:** M5.
- **Acceptance:** differential tests on normalized raised-error message text (which *requires* the printer) and handler control flow.
- **Risk:** You cannot match Racket's error text without the value printer — hence it is a hard dependency of this milestone, and it is also an explicit B2 deliverable.

### M7 — Full kernel (expander) primitive surface, data-driven **[XL, incremental] · ~2.5–3.5 mo**  ·  [#99](https://github.com/pmatos/nora/issues/99)
- **Goal:** Implement the *complete* finite set of primitives the **expander** references (a distinct, smaller set than the `racket/base` runtime surface in M16).
- **Deliverables:** the remaining prims by frequency (of the 544 measured by R3) — strings/bytes/chars + `format` (127×), the `letrec`/varref mechanism (`unsafe-undefined` **729×** sentinel + read-guard; `variable-reference-from-unsafe?` **637×** ⇒ always `#f` = safe path; `variable-reference->instance`, `variable-reference-constant?`, `variable-reference?`), srcloc structs + `syntax-source/line/column/position/span`, weak collections, places/atomic as **no-ops or a single global box** (`unsafe-make-place-local`/`-ref`/`-set!` = one global box per key; `unsafe-start/end-atomic` = no-ops), in-memory string/bytes ports + a stdout port so `display`/`format` print. (The real filesystem reader is M14/M15, not here.)
- **First task = spike R3:** instrument the interpreter to log **every unbound free identifier** hit while instantiating `expander.rktl` → an exact finite worklist; then batch-implement + differential-test.
- **Depends-on:** M4, M5, M6.
- **Acceptance:** the unbound-id log is **empty** after instantiating `expander.rktl` (meaningful only under M8's eager link-time binding); each prim batch has differential tests.
- **Risk:** The long-tail *fidelity* (not mere presence) is the danger; the logging instrumentation turns "mysterious expander crash" into a burn-down list.

### M8 — Linklet instances + `#%linklet` runtime API + eager resolution **[L] · ~1.5–2 mo**  ·  [#100](https://github.com/pmatos/nora/issues/100)
- **Goal:** Real linklet instantiation/linking, the runtime API the expander calls back into, and **eager link-time free-variable resolution**.
- **Deliverables:** an `Instance` type = named table of shared, mutable, GC'd variable **cells**; generalize `visit(Linklet)` to bind grouped imports to import-instances' cells and produce an instance (not a last-value); the `#%linklet` primitives NORA implements natively (`compile-linklet` = capture the parsed FEP AST + compiled marker; `instantiate-linklet` = reuse NORA's own linklet path; `eval-linklet`, `recompile-linklet`, `make-instance`, `instance-variable-value`/`-set!`, `instance-variable-names`, `linklet?`); `primitive-table : symbol → hash`, `primitive-lookup`, `declare-primitive-module!`; primitives grouped into pseudo-modules (`#%kernel`, `#%paramz`, `#%unsafe`, `#%flfxnum`; `#%foreign`/`#%network`/`#%place`/`#%futures` stubbed). **Design deliverable: instantiation eagerly binds every free reference to the primitive instance's cells** (as real linklets do), so M7/M9's "empty unbound-id log" is a complete proof, not a coverage artifact of executed paths.
- **Depends-on:** M7.
- **Acceptance:** a 2-linklet program (one exports, one imports a value) evaluates correctly; `(primitive-table '#%kernel)` returns a hash whose `car`/`cons` entries work; a linklet with an unbound free var **fails at instantiation**, not lazily at first use.
- **Risk:** Cross-instance references are mutable boxed cells shared by pointer — exactly what value-copy could not express; depends entirely on M2.

### M9 — Expander instantiates + `(boot)` **[L] · ~1–2 mo**  ·  [#101](https://github.com/pmatos/nora/issues/101)
- **Goal:** Run all ~2875 `define-values` and the boot tail to completion without an unbound/arity error.
- **Deliverables:** driver that instantiates the expander linklet into an instance exposing the **71 exports**, runs the boot tail (`namespace-init!`, `declare-reexporting-module!` loop for `#%kernel`/`#%paramz`/… and `#%linklet`/`#%boot`, `current-namespace ← ns`, `dynamic-require '#%kernel`), then fetches and calls the `boot` export. Replaces `main`'s "print last form" with an instantiate-then-boot driver; `runtime-instances` registration.
- **Depends-on:** M8; spike **R2** (measure cost).
- **Acceptance (fast CI smoke):** `expander.rktl` instantiates — all ~2875 closures built via flat capture, boot runs — with an **empty** unbound-id log (eager binding) and no arity error.
- **Risk:** Even with TCO+GC+flat closures, instantiating 2875 defines may be too slow/heavy (RSS/time). **Measure via R2 before committing to a full `expand` run;** if minutes/GBs, prioritize interpreter perf (persistent structures, allocation, closure layout — the exact thing M2's flat-closure decision de-risks).

### M10 — Self-hosted expand of a `#%kernel` expression **[M] · ~3–5 wk**  ·  [#102](https://github.com/pmatos/nora/issues/102)
- **Goal:** Call the expander's own `expand` on trivial input (self-hosted, no Racket oracle inside). `#%kernel` has **no macros**, so this exercises expansion without executing compile-time code.
- **Deliverables:** driver path `make-namespace` → set `current-namespace` → `namespace-require ''#%kernel` → build syntax via `datum->syntax` for `(module m '#%kernel (#%module-begin (display "hi")))` → `expand` → `syntax->datum`. No source loading, no user reader.
- **Depends-on:** M9, **M0-N (normalizer proven on real output)**.
- **Acceptance (expand-equivalence oracle):** NORA's `expand` output = `racket -e '(expand …)'` after **global-consistent** normalization.
- **Risk:** Hygienic gensym suffixes and hash-iteration order differ (R7); diff only after normalization, and fall back to observational equivalence where text cannot reconcile.

### M11 — Compile + eval a `#%kernel` module; bundle/directory model **[M–L] · ~4–6 wk**  ·  [#103](https://github.com/pmatos/nora/issues/103)
- **Goal:** End-to-end interpreted compile→instantiate of an expanded `#%kernel` module, modeling the **linklet-bundle/directory** structure explicitly.
- **Deliverables:** `compile` (expanded syntax → a **`linklet-directory` / `linklet-bundle`**, keys `decl`, `data`, `stx`, phase `0`/`1` bodies — *not* a monolithic single linklet) → phase-ordered instantiation (decl before body; syntax-literal/data linklets before phase-0 body) → `eval`/instantiate in the namespace → observe `"hi"` on stdout. **As soon as this exists, route NORA-produced FEP through the B-track compiler as a fixture** (see R8) so the two dialects integrate early, not at B5.
- **Depends-on:** M10.
- **Acceptance:** program output `"hi"` == Racket; exercises `compile-linklet`/`instantiate-linklet` and the bundle/directory instantiation order end to end.
- **Risk:** First real exercise of the compiled-linklet round-trip and the bundle structure in the runtime API.

### M12 — Phase-1 transformer execution (one-macro module) **[M–L] · ~1–1.5 mo**  ·  [#104](https://github.com/pmatos/nora/issues/104)
- **Goal:** Prove that NORA can **compile and instantiate a macro transformer at phase 1 and run it during expansion** — the single hardest mechanism, isolated *before* the full `racket/base` macro graph hits it.
- **Deliverables:** a tiny hand-written language exporting one macro (e.g. a `swap!`/`my-let` `define-syntax`); self-host-expand a module using it, which forces the expander to compile the transformer linklet, instantiate it at phase 1, and invoke it. Fed via `datum->syntax` (no reader/fs yet).
- **Depends-on:** M11.
- **Acceptance:** the one-macro module expands to the same normalized FEP as Racket; the transformer demonstrably executed inside NORA (traced instantiate at phase 1).
- **Risk:** This is the M10→"real hello" cliff made explicit. Everything in `racket/base` runs through this path ~thousands of times; discovering a phase-1 bug here is far cheaper than at M17.

### M13 — fasl v2/v3 + `#~` compiled reader + bundle deserializer **[L] · ~1.5–2 mo**  ·  [#105](https://github.com/pmatos/nora/issues/105)
- **Goal:** Load a machine-independent `.zo` `racket/base` graph. This is a large, finicky, **version-locked** format implementation, not "a bundle deserializer."
- **Deliverables:** the fasl v2/v3 reader (graph refs, srcloc/`prefab`/`path`/interned-symbol encodings), the `#~` compiled reader, `read-accept-compiled`, and the `linklet-directory`/`linklet-bundle` structure. A loaded `.zo` is fed through NORA's **own** `compile-linklet`/`instantiate-linklet` path (loading does not bypass the compiler).
- **Depends-on:** M11 (bundle model), **version-locked to `ORACLE_RACKET_COMMIT`**.
- **Acceptance:** a Racket-produced `.zo` bundle deserializes and its linklets instantiate under NORA, matching the oracle for a small module.
- **Risk:** Format drift vs. the pinned build silently corrupts everything; the fasl graph/`prefab`/`path` encodings are the finicky part.

### M14 — Minimal file I/O + embedded `racket/base` module table **[M–L] · ~1–1.5 mo**  ·  [#106](https://github.com/pmatos/nora/issues/106)
- **Goal:** Give the goal a real front door without a full OS layer. **Decision (committed now):** *embed the precompiled `racket/base` graph as an in-memory module table so no collection-path search is needed for the graph*, and provide only the minimal filesystem surface needed to read the **user's** `hello.rkt`.
- **Deliverables:** a real `path` datatype + `path->string`/`build-path`/`simplify-path` (minimal), `file-exists?`, `open-input-file` + UTF-8-decoding file input ports (for `hello.rkt`); an in-memory module registry preloaded with the M13 `racket/base` bundle graph, so `standard-module-name-resolver` resolves `racket/base` against the embedded table rather than the filesystem/collections.
- **Depends-on:** M13.
- **Acceptance:** `hello.rkt` reads as a UTF-8 char port; `(require racket/base)` resolves to the embedded graph with **zero** filesystem access.
- **Risk:** This decision changes M16's shape; making it *now* avoids building a collection-path search algorithm we do not need for hello.

### M15 — Reader + `#lang` / module-reader protocol **[L] · ~1.5–2 mo**  ·  [#107](https://github.com/pmatos/nora/issues/107)
- **Goal:** Real `read-syntax` for `#lang racket/base` over a file port. Reading `#lang` is the **language-loading protocol**, not `read` on s-expressions.
- **Deliverables:** the reader pipeline — read the `#lang` line → `read-language` → load the **reader module** (`syntax/module-reader`, itself macro-heavy, expanded/instantiated through M12's phase-1 path) → `read-syntax` producing syntax objects with **source locations** from the char port. `read` on datums, readtables to the extent `racket/base`'s reader needs. (`read-accept-compiled`/`#~` already landed in M13.)
- **Depends-on:** M14 (file ports), M16 partial or M13 (the `racket/base` reader module must be loadable). **The "`norac hello.rkt` reads" step lives here — it must not hide behind `datum->syntax`.**
- **Acceptance:** `read-syntax` on `hello.rkt` produces srcloc-bearing syntax whose `syntax->datum` matches Racket's, and whose `#lang` line correctly selects `racket/base`'s reader.
- **Risk:** The reader module is itself a program that must expand/instantiate under NORA — another exercise of M12's machinery.

### M16 — Module loader/resolver + `racket/base` graph + second primitive burn-down **[XL] · ~2.5–3.5 mo**  ·  [#108](https://github.com/pmatos/nora/issues/108)
- **Goal:** Instantiate the entire `racket/base` linklet graph and implement the **second, larger** primitive wave it needs at *runtime* (distinct from the expander surface in M7).
- **Deliverables:** module registry + **phase-ordered instantiation** (a module required `for-syntax` is instantiated at phase 1 before its requirer expands); `standard-module-name-resolver` + `current-load/use-compiled` wired against the M14 embedded table. **A second R3-style logging burn-down over `racket/base` *instantiation* (not expander instantiation)** — this surfaces primitives the expander never touches: the full keyword-application protocol (`make-keyword-procedure`, `keyword-apply`), the `for` runtime, more of the numeric tower (flonums, `bitwise-*`, `arithmetic-shift`), and the full print system. This wave is **comparable in size to M4–M7, not a footnote**, and is a first-class dependency of the goal.
- **Depends-on:** M12 (phase-1 macros execute), M13 (`.zo` loader), M14 (embedded table), M15 (reader).
- **Acceptance:** the `racket/base` graph instantiates with an **empty** second-wave unbound-id log; `(module m racket/base 42)` (read via M15 or fed via `datum->syntax`) self-host-expands and matches the oracle; `racket/base`'s macros execute through M12's phase-1 path.
- **Risk:** `racket/base` is a macro-heavy graph of dozens of modules — far heavier than `#%kernel`. Precompiled `.zo` avoids re-expanding it; **source-expansion of `racket/base` is the hardest possible bootstrap and is deferred** as a self-hosting-purity stretch goal.

### M17 — Interpreted `#lang racket/base` hello (Track A checkpoint) **[L/XL] · ~1–1.5 mo**  ·  [#109](https://github.com/pmatos/nora/issues/109)
- **Goal:** End-to-end self-hosted **interpretation** of a real `#lang racket/base` program. This is the integration crucible: it runs the full read → phase-1-macro-expand → compile → instantiate path over the entire `racket/base` graph.
- **Deliverables:** `norac hello.rkt` **reads** (M15), self-host-expands (NORA runs `expander.rktl`, `racket/base`'s transformers execute at phase 1 via M12), compiles to a bundle, and evals `(display "hello world")`.
- **Depends-on:** M16.
- **Acceptance:** `norac hello.rkt` stdout == `racket hello.rkt` stdout. **Track A complete; strongly-recommended checkpoint feeding B5.**
- **Risk:** Re-labeled **L/XL** (not M): it is the first time the entire compile+instantiate path runs over the entire `racket/base` macro graph. M12's one-macro proof de-risks it but does not eliminate the integration surface.

---

### B0 — Real NIR dialect skeleton (representation-agnostic) **[M] · ~1–1.5 mo** *(can start day one)*  ·  [#110](https://github.com/pmatos/nora/issues/110)
- **Goal:** Replace the broken scaffold with a real, round-trippable dialect — **without baking the value representation before M2 freezes it.**
- **Deliverables:** proper `NIR_Dialect` + **representation-agnostic** ops (`nir.const/quote`, `nir.if`+`nir.truthy`, `nir.app`/`nir.tailapp`, `nir.primcall`, `nir.lambda`/`nir.case_lambda`, `nir.linklet`/`nir.defvar`, `nir.box_new/ref/set`, `nir.let_values`/`nir.values`, `nir.wcm`) with verifiers and `CallOpInterface`/`SymbolOpInterface`/`RegionBranchOpInterface`; fix the duplicate/stale `.td`, class-name mismatch, and CMake wiring; a `noract` tool and `norac --emit=nir`. **Concrete value/unboxing types (`!nir.fixnum`, `!nir.box`, `!nir.closure`, interned `!nir.symbol`, struct descriptors) and unboxing bridges are deferred until after the M2 ABI freeze** (via R1), so lowering assumptions are not committed before the tag layout is decided.
- **Depends-on:** — for the ops; the concrete-type layer depends on R1/M2.
- **Acceptance:** `nir-opt` round-trips a `nir.constant`/`nir.return` `.mlir` (lit test); backend `mlir`-preset CI job (ccache, one compiler/config) builds.
- **Risk:** MLIR scaffolding is heavy before the first executable; keep MLIR out of the default matrix and out of the runtime/binary.

### B1 — Walking skeleton: FEP → NIR → LLVM → object → `./out` prints 42 **[M] · ~1–1.5 mo** *(highest-value early de-risk)*  ·  [#111](https://github.com/pmatos/nora/issues/111)
- **Goal:** Prove the *entire* toolchain on a trivial sublanguage (int literals, `+`, `if`) with a **throwaway** minimal value model — **no GC, no identity, no primitives.**
- **Deliverables:** FEP→NIR lowering (tail position computed here) + NIR→LLVM `TypeConverter`/ConversionPatterns → LLVM IR → `.o` via `addPassesToEmitFile`; a tiny runtime `main()`; a link step; `norac --emit=exe`.
- **Depends-on:** B0 (parallel with M1/M2).
- **Acceptance:** `norac --emit=exe tiny.rkt && ./out` prints the integer; == Racket oracle.
- **Risk:** Validates MLIR build + lowering + linking + runtime `main()` before broadening — classic end-to-end-first. **This is the only genuinely oracle-decoupled backend milestone.**

### B2 — Runtime library + ABI (`libnora_rt`), sharing Track A's primitives **[L] · ~2–3 mo**  ·  [#112](https://github.com/pmatos/nora/issues/112)
- **Goal:** The shared runtime built on the **frozen M2 value representation**, reusing the M4–M7 primitive library and the M6 printer — **not a second heap, not a second primitive set.**
- **Deliverables:** static `libnora_rt` = Boehm GC + `nr_value` allocation/constructors + **the kernel primitive library written once and shared with the interpreter** + **the value printer (shared with M6)** + symbol intern table + multiple-values buffer/ABI + arity/error helpers + `main()`/boot. Uniform code signature `nr_value code(nr_value env, i64 argc, nr_value* argv)`; closure = header + code-ptr + flat free-vars (byte-identical to M2's flat closure); MV = return-register fast path + thread-local values buffer.
- **Depends-on:** **M2 ABI freeze (hard gate)**, **M4–M7 (shared primitives + printer)**, B1.
- **Acceptance:** compiled `(let ([b (box 0)]) (set-box! b 5) (unbox b))` prints `5`; a compiled program that formats a value via the printer matches the oracle; runtime asan/lsan clean; triangulates `racket == norac-interp == norac-compiled`.
- **Risk:** Do **not** let A and B invent two heaps or two primitive sets — B2 consumes the M2 ABI and the Track-A primitive/printer library verbatim.

### B3 — Full FEP → NIR coverage **[L] · ~2–3 mo**  ·  [#113](https://github.com/pmatos/nora/issues/113)
- **Goal:** Compile every FEP form the interpreter supports.
- **Deliverables:** closure conversion (`make_closure` + top-level `code`), general `nir.app` + arity check, `case-lambda`, `values`/`let-values`, `letrec`, `set!`/boxes, structs (`make-struct-type` + props), vectors/hashes/strings, `#%variable-reference`, `with-continuation-mark` + mark queries, `error`/`raise`; **proper tail calls** via `tailcc` + `musttail` on the uniform signature (trampoline fallback behind a flag); prompts + one-shot escape continuations via the runtime. Racket-level opts in NIR (primcall inlining, fixnum unboxing, known-call devirtualization, dead-export elimination). **Full multi-shot `call/cc` deferred** — confirmed unnecessary for hello via R6.
- **Depends-on:** B2; spike **R6**; commit the continuation/mark ABI **early** (before this milestone) and share it with the interpreter.
- **Acceptance:** every interpreter-supported FEP form has a compile-and-run differential test; deep self/mutual tail loops run in O(1) native stack; **NORA-produced `#%kernel` FEP (from M11) compiles and runs**, not just oracle FEP.
- **Risk:** `musttail` target-portability (trampoline fallback); continuations-on-native-stack is the deepest backend problem — scoped to one-shot/prompts for hello.

### B4 — Linklet-bundle graph → NIR modules + runtime link/boot **[L] · ~1.5–2.5 mo**  ·  [#114](https://github.com/pmatos/nora/issues/114)
- **Goal:** Compile and link a multi-linklet **bundle/directory** graph into one executable.
- **Deliverables:** each linklet → an `instantiate(imports…) -> instance` function with top-level vars as heap boxes; **import resolution via runtime instance objects (not raw linker symbols)** — needed for mutable top-levels, re-instantiation, `dynamic-require`; boot instantiates the primitive instance + linklets in **dependency + phase order** (mirroring M11's bundle/directory instantiation exactly); wire the embedded `racket/base` graph deps.
- **Depends-on:** B3, **M8 (instance model), M11 (bundle/directory structure)**.
- **Acceptance:** a compiled multi-linklet hello graph links into one executable and runs; interpreted and compiled linking agree on instance semantics.
- **Risk:** Must mirror M8/M11 instance + bundle semantics exactly so interpreted and compiled linking agree.

### B5 — `./hello` from `#lang racket/base` (FINAL GOAL) **[L, convergence] · ~1–2 mo**  ·  [#115](https://github.com/pmatos/nora/issues/115)
- **Goal:** The standalone compiled executable.
- **Deliverables:** `hello.rkt` → NORA **reads + self-host-expands** (Track A: M15 + M17 path) → expanded linklet-bundle graph (+ embedded `racket/base` graph from M16) → NORA backend compiles (NIR→LLVM) → object files → link `libnora_rt` → `./hello`.
- **Depends-on:** M17 (self-hosted expander produces the FEP + `racket/base` graph) and B4 (compiler links a bundle graph).
- **Acceptance:** `./hello` prints `hello world`; stdout == `racket hello.rkt`. **Goal pipeline complete.**
- **Risk:** The convergence point; both tracks must land. Because M11 already routed NORA-FEP through the compiler (R8), the two dialects are integrated well before here. Continuations confirmed out of scope (R6).

---

## Critical path & honest parallelism

**Strict chain to the first `./hello`:**

```
M0 → M0-N → M1 → M2(freeze ABv)
                     │
   Track A (long pole, staff heaviest):
     M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M11 → M12 → M13 → M14 → M15 → M16 → M17
                     │                                  │
   Track B:          └── B0 → B1  (run ahead)           │
     B2 ⟵ needs M2 ABI + M4–M7 + M6 printer             │
     B3 ⟵ needs B2                                       │
     B4 ⟵ needs B3 + M8 + M11 (bundle model)            │
                                                         ▼
                              (M17) + (B4) ───────────► B5
```

- **Shared spine `M0 → M0-N → M1 → M2` blocks everything meaningful.** M2 is the fork and the ABI freeze; both tracks and the shared primitive/printer library consume it. One heap, one primitive set.
- **Only B0 and B1 run genuinely ahead.** B2 onward interleave with Track A: **B2 depends on M4–M7 + M6, B4 depends on M8 + M11.** Track B's *completion* is gated mid-Track-A. Staffing math: the two tracks are **not** fully parallelizable — plan for it.
- **The `racket/base` runtime primitive wave (M16) is a second cost center comparable to M4–M7**, on the critical path, and must not be conflated with the expander surface.
- **NORA-produced FEP flows through the compiler from M11 on** (R8), so B5 is a link-up, not a first meeting of two dialects.

## Top risks & early spikes (ordered by "hurts most if discovered late")

1. **R1 — Value model & GC ABI (highest; before M2 and before B0's concrete types).** Prototype tagged `nr_value` + Boehm heap + `eq?` identity + one mutable type + **flat closure capture**; prove a garbage-generating tail loop actually collects, and that the *same* representation plugs into both a trivial interpreter path **and** a trivial compiled program. **Commit Boehm-conservative for both interpreter and compiled code here** (moving/precise GC is out of scope, filed as a future XL). Getting this wrong rewrites both tracks.
2. **R7 — Hash-iteration-order & gensym-counter parity (threatens *every* M10→B5 differential test).** `eq-hash-code` is allocation-derived, so NORA's HAMT/`eq`-hash iteration order will differ from Racket CS; if the expander ever iterates a hash to emit code order, no alpha-normalizer can reconcile the result. Prove early that M5's hash iteration either matches Racket or is unused for output ordering; design M0-N for **global** consistent renaming; budget for an "observationally equivalent" acceptance fallback.
3. **R5 — MLIR/toolchain integration (parallel with M1/M2).** Ship **B1** (FEP→NIR→LLVM→`./out prints 42`) as early as possible to validate the MLIR build, lowering, linking, and runtime `main()` on one trivial case. Highest-value early Track-B de-risk.
4. **R3 — Primitive-surface long tail (two passes).** Instrument the interpreter to **log every unbound free identifier** — first while instantiating `expander.rktl` (first task in M7), then again while instantiating the `racket/base` graph (first task in M16). Converts two ~hundreds-of-prims unknowns into finite burn-down checklists; fidelity is caught by the eval-equivalence oracle. **Eager link-time binding (M8) is required for the log to be complete.**
5. **R2 — Expander cost (at M9).** Measure instantiate time + RSS of merely building all ~2875 closures **before** committing to a full `expand` run. If minutes/GBs, prioritize interpreter perf; the M2 flat-closure decision is the primary lever.
6. **R4 — `racket/base` provenance + fasl (before M13/M16).** Committed: **precompiled/embedded `.zo` graph** (fasl v2/v3 + `#~` + `read-accept-compiled`, version-locked to `ORACLE_RACKET_COMMIT`); source-expansion of `racket/base` deferred. Confirm the fasl `prefab`/`path`/graph encodings and the bundle/directory phase order early.
7. **R6 — Continuations in compiled code (before B3).** Confirm `#lang racket/base` hello does **not** exercise full multi-shot `call/cc` (near-certain), scoping compiled continuations to prompts + one-shot escapes. Commit the continuation/mark ABI early and share it with the interpreter.
8. **R8 — Two-dialect integration (at M11, not B5).** As soon as interpreted `#%kernel` compile exists, route **NORA-produced** FEP through the B-track compiler as a fixture, so shape/ordering differences between NORA-FEP and Racket-FEP surface early — not at the final milestone.
9. **R0 — TCO spike (M1).** Small: prototype `Call`-frame reuse + early frame-pop on a deep loop; assert bounded `Kont`.

## Effort realism

The S/M/L/XL scale saturates at the top; the honest picture is a **from-scratch Racket-CS bootstrap** where the incumbent (Chez) supplied GC, the numeric tower, structs, TCO, and continuations for free. NORA starts from a value-copy tree-walker with 6 primitives.

- **Cost centers (each multi-month, serial on the critical path):** **M2** (value model + GC + closures), **M5** (structs + hashes), **M7** (expander primitives), **M12** (phase-1 macro execution), **M16** (`racket/base` graph + second primitive wave), **M17** (integration crucible).
- The per-milestone person-time in each header is *individual* duration. **In strict series (this plan's own critical path), the interpreted hello (M17) is realistically 1–3 person-years before the backend is complete;** "months" applies to individual milestones, never the whole chain. Parallelism buys back only B0/B1 and modest overlap, because B2+ share Track A's substrate.
- Guardrails throughout: keep all ~90 integration tests green at every step (FileCheck catches any printed-form regression immediately), triangulate `racket == interp == compiled`, keep MLIR out of the default build, and use Racket purely as a differential oracle — never as the expander.

## Immediate next 3 actions (spikes)

1. **R1 — value-model/GC ABI spike (throwaway branch).** ([#88](https://github.com/pmatos/nora/issues/88)) **DONE** (commit `4e52ba6`): [`spike/r1-value-model/`](spike/r1-value-model/) proves a tree-walking interpreter and compiled C++ compute identical results over one shared heap/ABI, a flat closure applies identically in both, and Boehm churns **2.98 GiB of garbage through a 0.1 MiB live heap** (~24,000×, RSS 4 MiB; clean under `-O2` and ASan). The frozen tag/immediate/`ObjHeader`/flat-closure layout and the Boehm-conservative commitment (moving GC out of scope) are recorded in [`docs/value-model-abi.md`](docs/value-model-abi.md) — the reference M2 and B0/B2 consume.
2. **R3 (static pass) — unbound-identifier enumerator on today's tree.** ([#89](https://github.com/pmatos/nora/issues/89)) **DONE:** `tools/freevars.py` (scope-aware walker, robust to the M3 lexing gap) reports **544 distinct free identifiers** (22,173 refs) over 2,003 body forms; ranked, categorized worklist in [`docs/expander-primitive-surface.md`](docs/expander-primitive-surface.md). Seeds M4–M7.
3. **R5 — MLIR toolchain spike toward B1.** ([#90](https://github.com/pmatos/nora/issues/90)) Stand up the `mlir` preset against LLVM 22, replace the broken NIR scaffold with a minimal representation-agnostic `nir.constant`/`nir.return` dialect, and round-trip it through `nir-opt` — the first concrete step of B0 and the gate to the `./out prints 42` walking skeleton, de-risking the emit→link→`main()` chain before the ABI is even frozen.