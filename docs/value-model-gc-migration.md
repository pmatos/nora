All load-bearing facts verified against the tree: `Environment::Env` is `std::map<Identifier, shared_ptr<ValueNode>>` living inside `Scope` (a `make_shared` malloc block); `AllScopes` at `Interpreter.h:197` with teardown at `Interpreter.cpp:31-42`; `newScope` uses `make_shared` (`Interpreter.cpp:44`); `Box`/`Pair` are `shared_ptr<Cell>` over `unique_ptr<ValueNode>`; `Closure` holds `unique_ptr<Lambda>`; `Integer::~Integer()` at `AST.cpp:178`; `Symbol` uses `shared_ptr<void> Uninterned` + static `InternTable`; both test mains use `CATCH_CONFIG_MAIN`; `bdw-gc` 8.2.12 is present via pkg-config; the ABI's `nr_value` immediate layer is reusable but `nrt.h`'s `ObjHeader`-based object accessors are not. Here is the final plan.

---

# NORA M2 — Value model + Boehm-GC migration (final plan)

## 0. Purpose and hard promise

Migrate the interpreter's runtime values from a deep-copying, RAII, `shared_ptr`/`unique_ptr` object graph to a shared, Boehm-GC'd value model, converging structurally on the frozen R1 `nr_value` ABI. The whole test suite — both Catch2 unit mains (`test_interpreter`, `test_parse`) and the `lit`/FileCheck integration corpus — stays green under `debug`, `asan`, and `ubsan` at **every** slice, and **no slice ever leaves a GC cell embedding a destructor-bearing member, a live pointer into malloc-heap memory Boehm cannot scan, or a collectable value reachable only through an unscanned root.**

The honest shape of that promise: the *immediate* leaves (bool/char/fixnum/…) and the machine-plumbing refactors are genuinely small independent slices. The *mutually-recursive heap-object set* (`Box`, `Pair`, `List`, `Vector`, `Values`, `ContinuationMarkSet`, `Closure`) plus `Scope`/`Environment` form one strongly-connected component with **no valid one-at-a-time topological order** — a not-yet-migrated container can always hold an already-migrated element and vice-versa. We keep those slices individually green not by pretending an order exists, but by erecting explicit **transition scaffolding** (a legacy pin table + a GC keep-alive root) that makes every cross-tier reference safe, then demolishing it in the scope cutover. Several slices are behavior-preserving *characterization* refactors, not red→green; they are labeled as such and each carries an explicit characterization assertion.

---

## 1. Representation decision — (B): GC-backed `ValueNode`, staged toward the R1 `nr_value` ABI

Adopt **(B)**. Keep the `ValueNode` single-inheritance hierarchy and its virtual `accept`/`getKind` dispatch, but (i) allocate every runtime value with `GC_MALLOC` via placement-new, (ii) reduce every leaf/cell to pointer-free scalars and `nr_value` words — no `unique_ptr`/`shared_ptr`/`mpz_class`/`SmallString`/`SmallVector`/`std::map` members, (iii) make `clone()` an identity/no-op so shared references replace deep copies, and (iv) delete `AllScopes` and the teardown.

**Why not (A) (one-shot flatten to POD tagged `nr_value` words).** The value classes are welded to three subsystems that (A) would rewrite simultaneously in a single non-green cutover: the 27-way `ASTVisitor`/`dyn_cast` dispatch (`Runtime.cpp`, `valueEq`); the parser, where the *same* classes are AST literals *and* runtime values (`QuotedExpr` wraps a `ValueNode`, `Integer` is both a literal and the number type); and the tagged-immediate encoding. (B) changes **allocation and lifetime while keeping types and public API identical**, which is what keeps each slice green.

**Why the vtable is safe under Boehm.** The vptr is written by placement-new and points into `.rodata`; it is never a GC hazard and is never traced as a heap pointer. Boehm skips destructors, which is safe **iff** every member of a GC cell is trivially destructible or itself GC-managed — the invariant the ladder enforces. `ASTNode`'s base subobject (`const Kind` + `SMRange`, i.e. two `const char*`) is already trivially destructible, so a skipped virtual dtor on a POD-of-words leaf is genuinely safe.

**Convergence on the frozen ABI — and the crucial caveat.** Single inheritance keeps `(void*)this == (void*)base`, so the eventual flatten to `nr_value` is mechanical. **However, throughout all of M2 the GC cells remain polymorphic C++ objects: the vptr sits at offset 0, *not* an `ObjHeader{u32 type; u32 meta}`.** Reading `nr_obj(w)->type` on an M2 cell would read the low 32 bits of a vtable pointer — garbage. Therefore:

- M2 reuses **only the immediate/tag layer** of the frozen runtime: `nr_value`, `nr_fixnum`/`nr_fixnum_val`, `nr_bool`/`nr_truthy`, `nr_char`/`nr_char_val`, the `NR_*` singleton immediates, and the tag predicates. Heap pointers stay at tag `000` (a base pointer is its own `nr_value`).
- M2 **does not** use `nrt.h`'s object entry points (`nrt_cons`/`nrt_car`/`nrt_box`/`nrt_unbox`, `NrPair`/`NrBox`/`NrClosure`). Those `assert(nr_has_type(...))` on an `ObjHeader.type` that M2 cells do not have. They are dropped from the promoted header (or gated behind a `NORA_FLAT_ABI` macro) so nobody wires `nrt_unbox` onto a polymorphic cell and reads the vptr as a type tag. Object construction/access stays C++-RTTI (`getKind`/`dyn_cast`) for the whole milestone.
- The **flatten** — mapping the vptr slot onto `ObjHeader`, `getKind`→`NrObjType`, and swapping virtual dispatch for a `type`-`switch` — is a later, ABI/perf-driven step (post-M2, due only when compiled code (B2) shares the heap), **not** a GC-correctness requirement. `docs/value-model-abi.md` §6's "M2 consumes the ABI verbatim" is corrected to: *"M2 lands GC + shared references over the existing polymorphic hierarchy, reusing the `nr_value` immediate encoding; a later slice flattens the object layout to `ObjHeader`/`nr_value` for B2 heap-sharing."*

---

## 2. Invariants every slice preserves

**GC-cell ⇒ POD-of-words.** The instant a type is `GC_MALLOC`'d, its members may be only pointer-free scalars, raw GC pointers, and `nr_value` words. Two failure modes drive everything:

- **(L) Leak** — a skipped destructor never frees owned malloc heap (mpz limbs, `shared_ptr` control blocks, tree nodes). LSan does **not** catch this (Boehm's heap is invisible to it); the churn capstones (§5) do.
- **(D) Cross-heap dangle** — Boehm never scans malloc memory. A collectable GC value reachable *only* through malloc storage (a `unique_ptr`/`SmallVector` backing store, a `make_shared` `Scope`, an unscanned `std::map` node) is collected mid-evaluation → use-after-free.

**Two distinct rooting mechanisms, never conflated:**

1. **GC scanning** roots *migrated* (GC-cell) values. It works **only where the word physically lives is scanned by Boehm** — the C stack, or a container whose backing store is `gc_allocator`'d *and* whose header is itself reachable from a scanned location.
2. **A manual strong reference** roots *legacy* (malloc `ValueNode`) values regardless of where a referring word sits, because it keeps the object alive by identity, not by location.

**`Interpreter` stays stack-resident.** Its registers (`Val`, the `Kont` header, `Result`, `Env`) are scanned only because the object lives on the scanned C stack. A `make_unique<Interpreter>()` would move them to unscanned malloc and void the in-flight-value guarantee. A comment/`static_assert`-style note is added at its definition; if heap allocation is ever needed it must derive from `gc` (`gc_cpp.h`) or be registered as a root.

**Ordering law.** Immediates (never allocate) and interned symbols (`GC_MALLOC_UNCOLLECTABLE`, permanent roots) can be introduced freely. The **first *collectable* value** (String) may only appear once every place it can be parked across an allocation is covered by one of the two rooting mechanisms — which is exactly why the transition scaffolding (§3) is erected in the same slice.

---

## 3. The transition scaffolding (the migration vehicle)

A stack-only handle plus two temporary roots. Everything here is scaffolding that exists only for Phases 3–5 and is **deleted in S17/S18**.

```cpp
struct Value {          // lives ONLY on the C++ stack / Kont / env / GC-cell slots — a bare word
  nr_value W;           // immediate | GC pointer to a migrated cell | legacy-index immediate
};
```

`Value::W` is one of:
- an **immediate** (`nr_fixnum`, `nr_bool`, `nr_char`, `NR_NULL`/`NR_VOID`/`NR_EOF`/…);
- a **migrated** heap value: a GC pointer (tag `000`) to a polymorphic GC cell;
- a **legacy** value: a reserved-tag immediate (`NR_TAG_LEGACY`, the `0b100` slot the ABI marks "reserved") carrying an index into the legacy pin table. Because it is a non-pointer word, Boehm ignores it, so it is safe in *any* storage — a register, the malloc env map, or a migrated GC cell's slot.

**The two scaffolding roots (erected in S10, demolished in S17):**

- **Legacy pin table** — `std::deque<std::shared_ptr<ast::ValueNode>>`, an `Interpreter` member (a plain RAII container). It owns every still-malloc `ValueNode` and destroys each **exactly once** at interpreter teardown. This is the *manual strong reference* of §2: a legacy value referenced from anywhere — including a GC cell's word slot — stays alive and is freed once. It does **not** need to be GC-scanned.
- **GC keep-alive root** — `std::vector<nr_value, gc_allocator<nr_value>>`, an `Interpreter` member. Its header is on the scanned stack, so Boehm traces its buffer. Whenever a **migrated** (GC-pointer) word is written into storage Boehm cannot scan — a still-legacy container's slot, or the still-malloc env map — that word is appended here. This is the *GC scanning* mechanism of §2, extended to reach malloc-resident words. It holds no destructors and is reclaimed wholesale (its cells too) once the scaffolding is gone.

Together these dissolve the SCC's two hazard directions: a **migrated cell holding a legacy element** is safe (legacy-index word + pin table), and a **legacy container (or malloc env map) holding a migrated element** is safe (keep-alive root). This is what lets each container flip to a GC cell as its own individually-green slice with no valid topological order and no `intoCell`-to-word bridge for unmigrated types (which cannot exist).

**Cost, and why it is acceptable.** The scaffolding over-retains: legacy values and any migrated value that ever entered malloc storage live until teardown, so memory is O(work) during Phases 3–5. That is correct (no UAF, no double-free, no LSan leak — everything is freed once at teardown), and it does not pollute the capstones, which are deliberately written over already-migrated, non-parked value garbage (§5). The final slices delete the scaffolding, at which point memory is bounded by the GC.

In Phases 1–2, before any GC cell can hold a `Value`, the legacy alternative is simply materialized at the boundary (a fresh RAII `ValueNode` *view* owned by the still-legacy container, or a `shared_ptr` inside the stack-resident `Value`); the pin-index encoding is switched on in S10 when the first GC cell that can hold an arbitrary `Value` approaches.

---

## 4. Slice ladder

Legend: **RED** = a genuine failing test drives the slice; **CHAR** = behavior-preserving refactor guarded by an explicit characterization assertion.

### Phase 0 — Collector up; scan the roots that are already reachable

**S0 — libgc linked, `GC_INIT`, heap hooks. (RED — link failure.)** *Detailed in §7.*
Promote `spike/r1-value-model/nrt.{h,cpp}` → `src/nora_rt.{h,cpp}`, **stripping/gating the `ObjHeader`-based object accessors** (§1) and **removing `GC_set_all_interior_pointers(1)`** from `nrt_init`. Wire `PkgConfig::BDWGC` into `src/CMakeLists.txt` and both test exes; `GC_INIT()` first in `main`; convert both test mains to `CATCH_CONFIG_RUNNER`; expose `getGCHeapSize()`/`getGCTotalBytes()`. RED: `nr_*`/`GC_*` unresolved. Depends-on: —.

**S1 — `gc_allocator` the containers whose headers are already scanned. (CHAR.)**
`Kont` (`std::vector<Frame>`, `Interpreter` member → header on the stack), `Frame::Done`, `Frame::Marks`, and `ContinuationMarkSet::Frames` (which are traced transitively once they sit inside the scanned `Kont` buffer). Element *types* are unchanged (still `unique_ptr`/`shared_ptr`); only the allocator changes, so the buffers are now scanned while elements are still RAII-destructed normally by their stack-resident owners.
**The `Environment` map is deliberately NOT touched here.** Its header lives inside a `Scope` created by `make_shared` (malloc, `Interpreter.cpp:44`); `gc_allocator`-ing its nodes would move the red-black-tree nodes to GC memory whose only inbound edge (`std::map::_M_header`) lives in unscanned malloc — an unrooted collectable component that `gc_allocator`'s own `GC_MALLOC` could reclaim mid-run → UAF in `envLookup`. The env map becomes GC storage only in S17, in the same slice its owning `Scope` becomes a scanned GC cell. Characterization: full suite + a debug assert that `Kont.data()` is a GC pointer. `GC_add_roots` is rejected (vectors relocate on growth); interior-pointer scanning is rejected (§6, R6). Depends-on: S0.

### Phase 1 — Machine speaks `Value` (100% legacy inside)

**S2 — `Value` in registers. (CHAR.)** `Val`, `Result`, `getResult()` internals → `Value`; legacy alternative held RAII inside the stack-resident handle. Assertion: whole suite; a legacy `Value` is behaviorally identical to today's `unique_ptr`. Depends-on: S1.

**S3 — `Value` in `Environment`; kill clone-on-lookup. (CHAR + new aliasing test.)** `add`/`lookup`/`envLookup`/`envSet` store and return `Value`; lookup **shares** instead of `clone()`ing. Sharing an immutable value equals cloning it; mutable values (`Box`/`Pair`) already share via their inner cell. Audit (must all be pinned by an assertion): `+`/`-`/`*` build fresh accumulators; `SubtractFunction` clones the first arg before `-=` (`Runtime.cpp:52`); no primitive mutates a looked-up value in place; **argument aliasing** — `(f x x)` now passes one shared object twice (correct for Racket — pin it); the WCM key/value paths. Standing guard adopted here for the *rest of the milestone*: at the moment **any** type becomes shared, add a "mutate through one reference, observe through another" test (template: the existing `set-box!`/`set-car!` tests). Depends-on: S2.

**S4 — `Value` in frames. (CHAR.)** `Frame::{Done,Saved,Callee,WcmKeyV}` and `MarkFrame` entries → `Value`. Depends-on: S3.

**S5 — Test-seam helpers. (CHAR/refactor.)** Add `expectInt(Run,42)`, `expectBool(Run,true)`, `expectResult(...)` wrapping today's `dyn_cast<ast::Integer>` assertions in `test/unit/test_interpreter.cpp`, localizing the eventual seam flip to one place. Depends-on: S2.

### Phase 2 — Immediates (no allocation, no roots)

**S6 — Booleans → `NR_TRUE`/`NR_FALSE`. (CHAR; forcing: `#f` result `== NR_FALSE`.)** `IfBranch` reads truthiness via `nr_truthy`; `getResult()`/`write()` materialize a `BooleanLiteral` view at the public seam. Depends-on: S4, S5.

**S7 — Char / Void / Null / eof → immediates. (CHAR.)** Same pattern; retires `Char`'s `SmallString<8>` (`AST.h:435`). Guarded by char/void `.rkt` tests + the seam shims. Depends-on: S6.

**S8 — Fixnums → `nr_fixnum`. (RED — arithmetic/overflow.)** In-range `Integer` becomes an immediate; `+`,`-`,`*`,`zero?` get a word fast path with `__builtin_*_overflow`; **overflow promotes to a still-legacy `ast::Integer` bignum** (unchanged malloc object via the legacy path). RED driver: the deep tail-loop now runs on immediate arithmetic, plus an overflow-promotion test. Depends-on: S7.

### Phase 3 — Erect the scaffolding; collectable leaves

**S9 — Symbol → interned `GC_MALLOC_UNCOLLECTABLE` cell. (CHAR + identity tests.)** Name via `GC_MALLOC_ATOMIC_UNCOLLECTABLE`; `eq?` = pointer identity. Deletes `Symbol::Uninterned` (`AST.h:235`) and the static `unordered_set InternTable` (`AST.cpp:96`); `gensym`/`string->uninterned-symbol` = a distinct GC symbol whose address is its identity. Interned symbols are *permanent, uncollectable roots*, so binding one anywhere needs no env scanning and nothing leaks (Boehm's uncollectable list roots them; LSan cannot see them). Guarded by `symbol eq? is identity`, `gensym`, and uninterned tests. Depends-on: S8.

**S10 — Erect the scaffolding + String/Keyword → GC-atomic cells. (RED — `eq?`-on-string + GC-survival.)** Switch the legacy `Value` representation to the pin-index encoding and stand up the **legacy pin table** and the **GC keep-alive root** (§3). Make `String`/`Keyword` `ObjHeader`-less GC cells (`len` + `GC_MALLOC_ATOMIC` bytes), retiring their `SmallString`s. **This is the first *collectable* value:** every collectable word written into the still-malloc env map or a still-legacy container is now registered in the keep-alive root, so it survives a collection. RED drivers: a new `eq?`-on-string test documenting the intended post-migration pointer identity (NORA currently routes `String` through *structural* `valueEq`, so the flip is otherwise invisible), and a "bind a string, run a GC-forcing expression, read the string back" test under asan. Depends-on: S9, S1.

**S11 — Bignum `Integer` → GC cell over GC-atomic `mpz_t`; install the GMP hook. (RED — bignum-through-GC + `test_parse` stays leak-clean.)** In one slice: make `Integer` a `GC_MALLOC` (scanned) cell holding a **raw `mpz_t`**, drop `~Integer`/`mpz_clear` (`AST.cpp:178`), and install `mp_set_memory_functions(gmp_alloc, gmp_realloc, gmp_free)` routing limbs through `GC_MALLOC_ATOMIC`, with the custom free being **`GC_FREE` or a no-op — never system `free`** (an `mpz` allocated before the hook and freed after would otherwise cross allocators and crash). The hook is process-global and must be installed **before the first GMP allocation in all three entry points** — `norac` `main` **and both** `CATCH_CONFIG_RUNNER` test mains, including `test_parse`, which constructs `Integer` literals at parse time and links `gmpxx`; if its main omits the hook, dropping `mpz_clear` leaks limbs and `test_parse` (which does not disable LSan) turns red. Verify no `static`/global `Integer` and no `gmpxx` temporary predates the install. Bundling is mandatory: installing the hook while `Integer` is still a malloc object would let Boehm collect limbs referenced only from unscanned memory (D). The scanned cell follows the (atomic, unscanned) limb pointer, so no false retention. Fold in the remaining leaves (`RuntimeFunction` index, `VariableReference`) as POD cells/immediates. After S11 **every leaf is a word.** Depends-on: S10.

### Phase 4 — Containers become GC cells (each individually green via the scaffolding)

Every container flips to a `GC_MALLOC`'d **polymorphic** C++ cell (vptr at offset 0, safe under Boehm; *not* `nrt.h`'s `NrPair`/`NrBox` — §1) whose slots are `nr_value` words. Cross-tier references are safe by construction: legacy elements ride the pin-index + pin table; migrated words stored into any still-legacy container ride the keep-alive root. No GC cell embeds a `unique_ptr`/`shared_ptr`/`mpz`/`SmallString`.

**S12 — Box → GC cell. (CHAR + `set-box!` identity.)** Single `nr_value` slot; retires `shared_ptr<Cell>` (`ASTRuntime.h:93`). Shared identity falls out of pointer identity. Depends-on: S11.

**S13 — Pair → GC cell. (RED — value-garbage capstone, §5.)** The hot allocation site; retires `shared_ptr<Cell>` (`ASTRuntime.h:127`). The capstone loop churns *transient, unbound* `cons`/`box` garbage while binding **only immediates** (`n`, `acc` are fixnums), so nothing it produces is parked in the env or a legacy container and thus nothing is held by the keep-alive root — the garbage is genuinely collectable and the GC heap plateaus. RED before this slice (malloc `shared_ptr` cells → GC churn ≈ 0), GREEN after. Guarded also by `set-car!`/`set-cdr!` identity tests. Depends-on: S12.

**S14 — Values / List / Vector → variable-length GC cells. (CHAR.)** Length in a POD field; elements already words. Retires their `SmallVector`s. `quote8.rkt`'s `'((1 2 3) #("z" x) . the-end)` — a `List` of a `List` and a `Vector`-of-`String`+`Symbol` — crosses tiers freely and stays green because the scaffolding roots both directions. Depends-on: S13.

**S15 — ContinuationMarkSet / MarkFrame → GC cells. (CHAR + WCM-across-GC test.)** Elements are words. Guard: a `with-continuation-mark` whose result expression forces a collection, then the mark is read back (R4; the existing `with-continuation-mark{1..6}.rkt` do not force a collection). Depends-on: S14.

### Phase 5 — Closures, the scope cutover, demolish the scaffolding

**S16 — Closure / CaseLambdaClosure → flat GC cells. (CHAR + GC-mid-loop correctness.)** Captures are words; the body points at the **shared, immortal AST `Lambda`** owned by `main`'s `unique_ptr<Linklet>` (which outlives all evaluation), retiring `unique_ptr<Lambda>` (`ASTRuntime.h:36`) and `unique_ptr<CaseLambda>` (`ASTRuntime.h:58`). This makes `Frame::Call::Callee` no longer load-bearing for `Control`-into-body validity, so its slot is dropped. **No intermediate may exist where the closure is a GC cell while its body is still a per-closure `new`'d clone** — Boehm never scans that clone, so a `const ASTNode* Control` interior pointer into it would dangle on collection (and the clone would leak, dtor skipped). Point at the shared AST in the *same* slice that GC-allocates the closure. Guards: peak-Kont tests (`Deep == Shallow < 16`) stay green; a mutual-tail-recursion loop that forces a GC mid-loop and asserts the result, under asan+LSan. Depends-on: S15.

**S17 — The scope cutover (coupled by design) + demolish the scaffolding. (RED — scope-garbage capstone.)** This slice is *deliberately* one atomic step because its parts are interlocked:
- `Scope` → GC cell; `Parent` becomes a raw GC pointer.
- `Environment`'s map nodes → `gc_allocator`. This is now safe (unlike S1): the GC-cell `Scope` holding the map header is itself scanned, so the header, the tree nodes, and their `Value` words are all traced. The mapped type is a POD `nr_value` word and the key `ast::Identifier` is trivially destructible, so **a GC `Scope` whose `~map` never runs leaks nothing** — the dtor-leak trap is closed by construction, not by requiring "all values migrated first."
- Atomically delete `AllScopes` (`Interpreter.h:197`), `newScope`'s accumulation (`Interpreter.cpp:47`), and `~Interpreter`'s cycle-break (`Interpreter.cpp:31-42`). Going straight to `GC_MALLOC` (not `allocate_shared`) avoids the split-brain where a `shared_ptr` scope is reachable only through the malloc `AllScopes` buffer that Boehm frees under it. Boehm now reclaims the closure↔scope cycles that the teardown used to sever by hand.
- **Demolish the scaffolding:** every value type is now a word or GC cell, so the legacy pin table is empty and is deleted; the env map and all containers are now scanned, so the GC keep-alive root is no longer needed and is deleted.
- Add `"environment": { "ASAN_OPTIONS": "detect_leaks=0" }` to the `asan`/`ubsan` `testPresets` (the GC heap is now intentionally never torn down).

RED driver: the scope-garbage capstone (§5) — a deep tail loop creating one scope per iteration — shows O(depth) heap while `AllScopes` roots every scope, and a bounded, depth-independent heap once scopes are GC and `AllScopes` is gone. Depends-on: S16.

**S18 — Collapse the seam; delete `clone()`. (CHAR/refactor, suite updated same commit.)** Rewrite the S5 helpers and `getResult()`/`write()` to the `nr_value` seam; delete the `Value` legacy path and pin-index materialization, `ValueNode::clone()`, and every `ClonableNode` override. End state: `clone()`/`AllScopes`/scaffolding all gone, fully GC'd value model over the reused `nr_value` immediate encoding. Depends-on: S17.

---

## 5. The forcing GC-heap seam — heap-size hook, not RSS

Expose `getGCHeapSize()`/`getGCTotalBytes()` (`GC_get_heap_size`/`GC_get_total_bytes`) at the interpreter unit seam (mirroring the existing `getPeakKont()` at `Interpreter.h:81`) and assert a **depth-independent live-heap plateau against unbounded churn**. RSS is rejected: it is a monotonic high-water peak (cannot observe in-process reclamation), is polluted by the LLVM/binary footprint and glibc arenas, is meaningless under ASan shadow memory (so it cannot run in the sanitizer presets), and is contaminated by sibling Catch tests in one process. The GC-heap ratio is deterministic, preset-robust, and isolated to GC bytes.

**Two distinct capstones, each scoped to the garbage it can actually observe:**

*Value-garbage capstone (S13).* The loop churns transient, **unbound** `cons`/`box` garbage while binding only immediates, so the scaffolding's keep-alive root never retains it — it is genuinely collectable. Because the env map and scopes are still malloc through Phase 4 (and thus invisible to `getGCHeapSize`), this capstone measures **value** garbage only.

*Scope-garbage capstone (S17).* A deep tail loop creating one scope per iteration; asserts `getGCHeapSize()` bounded and depth-independent once scopes are GC and `AllScopes` is deleted.

This resolves the earlier heap-accounting contradiction: during Phases 3–5 the scaffolding intentionally holds env-bound and parked values (memory is O(work) by design), so the value capstone must **not** assert a globally bounded heap — only that its own transient garbage is reclaimed. Global bounded memory is asserted only at the scope capstone.

**Robust assertion shape** (avoid magic-number flake):

```cpp
static size_t liveHeapAfter(long depth) {
  Interpreter I(Diag);
  (void)GC_malloc(1 << 16);        // warm-up: don't let the initial heap block dominate
  AST_for(depth)->accept(I);
  GC_gcollect();                   // force a collection before sampling
  return I.getGCHeapSize();
}
// Assert a PLATEAU across two LARGE depths, holding machine-structure constant,
// rather than anchoring to a small baseline:
auto hA = liveHeapAfter(1'000'000);
auto hB = liveHeapAfter(8'000'000);   // 8x the work
REQUIRE(hB < hA * 2);                 // 8x work, < 2x live heap  => reclamation happened
```

Keep the existing `norac` end-to-end deep-loop `.rkt`/FileCheck as a coarse no-OOM backstop.

---

## 6. Risks and guards

- **R1 — Unscanned parking storage → mid-eval UAF.** In-flight registers are stack-scanned, but values parked between steps live in container storage Boehm never sees. *Guard:* S1 scans the stack-header-rooted containers (`Kont`/`Done`/`Marks`/CMS); the **env map is not scanned until S17** (scanning its nodes while its header is in malloc is itself the UAF — see S1); collectable values parked in still-malloc storage between S10 and S17 are held by the GC keep-alive root. Plus a GC-forcing many-argument `App`/`values` correctness test under asan.
- **R2 — Closure body vs. `Control` interior pointer.** *Guard:* S16 repoints closures at the shared immortal AST in the same slice it GC-allocates them — never a mixed intermediate; mutual-tail-recursion GC-mid-loop test under asan+LSan.
- **R3 — Scope-GC / `AllScopes` deletion interlock.** Removing `AllScopes` before scopes are GC reintroduces the `shared_ptr` cycle leak; GC-allocating a `Scope` whose map still held `shared_ptr` values would leak. *Guard:* S17 does all of it atomically, and the POD `Value`-word map makes a GC `Scope` dtor-free.
- **R4 — Continuation-mark value collected from an unscanned frame.** *Guard:* S15 GC-cells the marks; a set-mark → GC-forcing-result → read-mark test.
- **R5 — Silent `eq?`/identity drift when a type moves from clone to shared.** *Guard:* the S10 `eq?`-on-string test pins the intended identity before the flip; the standing "mutate through one reference, observe through another" guard is applied to **every** type at the slice it becomes shared, and the S3 audit explicitly covers argument aliasing (`(f x x)`) and the WCM key/value paths.
- **R6 — `GC_set_all_interior_pointers(1)` (spike default) worsens false retention** and never makes malloc memory scanned. Heap pointers stay at tag `000`; `Control` interior pointers are into the *non-GC* immortal AST. *Guard:* S0 drops it.
- **R7 — Heap-allocated `Interpreter` unscans the registers.** *Guard:* keep it stack-resident (it already is everywhere); note at its definition.
- **R8 — Half-migrated destructor leak (a GC cell embedding `unique_ptr`/`mpz`/`shared_ptr`), invisible to LSan.** *Guard:* the POD-of-words invariant, the pin/keep-alive scaffolding (so a GC cell never embeds a legacy owner), and the S13/S17 `GC_gcollect()` heap-ratio tests, which surface any residual leak as an unbounded live heap. Reviewers verify specifically S11 (drop `mpz_clear` only with the GMP-hook flip, in all three mains, with a GC-safe free) and S16 (drop `unique_ptr<Lambda>` only with the shared-AST repoint).
- **R9 — Boehm/LLVM signal-handler and threading interaction.** `GC_INIT()` runs before `llvm::InitLLVM` (which installs signal handlers). Keep Boehm **non-incremental** (the default) so there is no `SIGSEGV` dirty-bit handler for LLVM to clobber; document this constraint (it also bears on the future `call/cc` note).
- **Future (out of scope).** `call/cc`/delimited continuations may introduce real C++ exceptions or stack copying, reopening Boehm↔unwinder and stack-scanning questions. Today the eval path has no exceptions (errors go through `Diag.error` + `abortEval`'s manual `Kont.erase`), which is GC-friendly.

---

## First slice: do this now — S0

**Failing test first** (`test/unit/test_gc.cpp`; it exercises only the reusable *immediate* ABI layer and the collector — never a polymorphic-cell object accessor):

```cpp
#include <catch2/catch.hpp>
#include "nora_rt.h"
#include <gc.h>

TEST_CASE("libgc links, inits, and the nr_value immediate ABI is usable", "[m2][gc]") {
  REQUIRE(nr_fixnum_val(nr_fixnum(42)) == 42);   // RED: nora_rt / libgc not linked yet
  REQUIRE(nr_truthy(nr_bool(true)));
  REQUIRE_FALSE(nr_truthy(NR_FALSE));
  void *p = GC_MALLOC(64);                        // exercise the collector
  REQUIRE(p != nullptr);
  REQUIRE(GC_get_heap_size() > 0);
}
```

New shared test main (both existing mains switch to it):

```cpp
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <gc.h>
int main(int argc, char **argv) {
  GC_INIT();                        // records the main-thread stack bottom; GMP hook is S11
  return Catch::Session().run(argc, argv);
}
```

**Minimal implementation:**

1. **Promote the runtime.** `git mv spike/r1-value-model/nrt.{h,cpp}` → `src/nora_rt.{h,cpp}`, keeping the `nr_*` immediate/tag layer. **Strip or `#ifdef NORA_FLAT_ABI`-gate the `ObjHeader`-based object entry points** (`nrt_cons`/`nrt_car`/`nrt_box`/`nrt_unbox`, `NrPair`/`NrBox`/`NrClosure`) so they cannot be wired onto M2's polymorphic cells. In `nrt_init`, **remove `GC_set_all_interior_pointers(1)`**.
2. **CMake.** In the root `CMakeLists.txt` (beside the LLVM/GMP `find_package`):
   ```cmake
   find_package(PkgConfig REQUIRED)
   pkg_check_modules(BDWGC REQUIRED IMPORTED_TARGET bdw-gc)   # verified: bdw-gc 8.2.12 present
   ```
   `src/CMakeLists.txt` — add `PkgConfig::BDWGC` to `LIBS` and add `nora_rt.cpp` to the `norac` sources. `test/unit/CMakeLists.txt` — add `${PROJECT_SOURCE_DIR}/src/nora_rt.cpp` to **both** exes and append `PkgConfig::BDWGC` to both `target_link_libraries` (`test_parse` compiles `AST.cpp`, which routes GMP through GC in S11) and add `test_gc.cpp` to the `test_interpreter` sources.
3. **`GC_INIT()` placement.** `src/main.cpp` — first statement of `main()`, before `llvm::InitLLVM` and `Parse::parseLinklet`. Convert `test/unit/test_interpreter.cpp:1` and `test/unit/test_parse.cpp:2` from `CATCH_CONFIG_MAIN` to `CATCH_CONFIG_RUNNER` with the shared `main` above, so `GC_INIT()` runs on the main thread before any test allocates. (The GMP hook is **not** added here — it lands in S11 in all three mains.)
4. **Heap hooks** on `Interpreter` (test-only, beside `getPeakKont()`):
   ```cpp
   size_t getGCHeapSize()   const { return GC_get_heap_size(); }
   size_t getGCTotalBytes() const { return GC_get_total_bytes(); }
   ```
5. **CI.** Add `libgc-dev` to the apt install lists in `.github/workflows/`. No new preset — GC is a hard dependency of the default build (unlike opt-in MLIR), inherited by every preset via `LIBS`.

**Acceptance:** the whole suite (both Catch2 mains + the integration corpus) green under `debug`/`asan`/`ubsan`. If LSan reports Boehm-internal allocations under `asan`, add `ASAN_OPTIONS=detect_leaks=0` to the `asan`/`ubsan` `testPresets` immediately; otherwise defer that flag to S17. Proving Boehm ↔ sanitizer ↔ `InitLLVM` coexistence here, before any value depends on the collector, is the entire point of doing S0 first.