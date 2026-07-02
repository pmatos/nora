# NORA value-model ABI (frozen by spike R1)

This is the shared runtime representation of Racket values in NORA: one tagged
word, one heap-object header, one closure layout, and one garbage collector,
used by **both** the tree-walking interpreter (milestone M2) and statically
compiled code (B0/B2). Freezing it once is the whole point of spike R1 — the
interpreter and the compiler must agree byte-for-byte or they cannot share a
heap, and changing it later forces rewriting both tracks.

The design below is validated by the runnable spike in
[`spike/r1-value-model/`](../spike/r1-value-model/) (issue #88). Numbers quoted
here are what the spike prints on x86-64 with Boehm GC 8.2.

## 1. `nr_value`: a tagged 64-bit word

```
  bit 0 == 1                      fixnum        value = (int64)w >> 1   (63-bit)
  low 3 bits == 0b000 (w != 0)    heap pointer  an 8-byte-aligned ObjHeader*
  low 3 bits == 0b010             singleton     subtype = w >> 3
  low 3 bits == 0b110             character     codepoint = w >> 3
  low 3 bits == 0b100             reserved
```

Rationale:

- **Fixnums are odd** (`(v << 1) | 1`). Arithmetic (`+`, `-`, `<`) works on the
  raw word with a single untag/retag and an `__builtin_*_overflow` check; the
  overflow path promotes to a bignum (M4 — the spike asserts instead).
- **Heap pointers keep tag `000`**, so an `ObjHeader*` *is* its own `nr_value`
  with no masking on dereference — the common operation stays free. Every heap
  object is 8-byte aligned (Boehm returns ≥16-byte-aligned granules).
- **Singletons and chars are even and non-zero-low-3**, so they never collide
  with odd fixnums or `000` pointers.

Singleton subtypes (`(subtype << 3) | 0b010`): `#f`, `#t`, `'()`, `void`, `eof`,
`unsafe-undefined` (the 729×-referenced sentinel), and a letrec `uninit` hole.
Racket truthiness is `w != #f` — everything else, including `0` and `'()`, is
truthy.

Reserved tag `100` is left for a future need (e.g. an immediate flonum on 64-bit
if we ever want NaN-free unboxed doubles); nothing depends on it today.

## 2. Heap objects: uniform `ObjHeader`

Every heap object starts with an 8-byte header so payloads stay 8-aligned:

```c
struct ObjHeader { uint32_t type; uint32_t meta; };   // type = NrObjType, meta = len/arity/flags
```

Spike object sizes: `NrPair` 24 B (`header + car + cdr`), `NrBox` 16 B,
`NrClosure` 24 B + captures, `NrSymbol` 24 B. The header is deliberately fat
enough (`meta`) to carry a length/arity without a second word; a future precise
GC would use spare header bits for mark/forward state (see §5).

## 3. Closures: flat capture, one code signature

```c
typedef nr_value (*nr_code)(nr_value self, int64_t argc, const nr_value *argv);
struct NrClosure { ObjHeader h; nr_code code; uint32_t nfree, pad; nr_value free[]; };
```

A closure is `header + code-pointer + inline captured cells`. Capture is
**flat**: a lambda closing over the ~2000-entry linklet top level captures only
the variables it uses, so building it is O(captured), not O(env) — the property
that de-risks instantiating the expander's thousands of closures (M9/R2).

`nr_code`'s signature is the exact one B2's compiled code uses: the callee
receives its own closure as `self` and reads captures from `free[]`. So the same
`nrt_apply(clos, argc, argv)` entry point dispatches an interpreter-built
closure and a compiler-emitted one identically — proven in the spike, where a
closure capturing `10` applied to `5` yields `15` through both paths.

## 4. Garbage collection: Boehm conservative — committed for both tracks

NORA uses the **Boehm-Demers-Weiser conservative collector** (`libgc`) for the
interpreter **and** compiled code, sharing one heap. A conservative collector
scans the native C/C++ stack and registers, which is exactly what both the CEK
machine's registers (`Kont`/`Env`/`Val`) and compiled `tailcc` frames need — no
stack maps, no shadow stack, no write barriers to bring up first.

Spike evidence: a loop allocating a box and a pair per iteration for 50 M
iterations **churns 2.98 GiB through a 0.1 MiB live heap (≈24,000×)** with peak
RSS 4 MiB — Boehm reclaims the per-iteration garbage and the heap stays bounded.
The interpreter and compiled paths behave identically.

**Out of scope for the hello-world goal (filed, not planned):** a moving/precise
GC (LLVM statepoints / shadow stack). It would change this ABI (object headers,
interior-pointer rules, `musttail` interaction) and is an XL effort of its own.
There is **no** "reserved forwarding slot" pretending to future-proof the ABI —
if precise GC is ever adopted it is an explicit ABI break.

## 5. Known conservative-GC caveats (accepted)

- **False retention from fixnum bit patterns.** An odd fixnum word can, by
  coincidence, look like an interior pointer into a live object, briefly keeping
  garbage alive. This is a correctness-preserving imprecision inherent to
  conservative GC; the spike shows it does not stop the heap from staying
  bounded. It is one motivation for the (out-of-scope) precise-GC future.
- **GMP.** Bignums must route their limb storage through GC-atomic allocation
  (pointer-free) and drop `mpz_class` RAII for GC-managed raw `mpz_t`, so bignum
  `nr_value`s need no finalizers (M2/M4 task).
- **Non-GC containers.** Anything holding the only reference to a live object
  from malloc'd memory the GC does not scan (e.g. a `std::unordered_map` intern
  table) must either be a registered root or hold *uncollectable* objects. The
  spike interns symbols with `GC_MALLOC_UNCOLLECTABLE` (symbols are permanent).

## 6. What M2 / B0 / B2 must consume verbatim

1. The tag scheme and immediate encodings in §1 (`nrt.h` is the reference).
2. The `ObjHeader` shape and 8-alignment in §2.
3. The flat-closure layout and `nr_code` signature in §3 — interpreter closures
   and compiled closures are the same objects.
4. Boehm conservative GC as the single heap for both tracks (§4).

The spike's `nrt.h`/`nrt.cpp` are the throwaway proof; M2 promotes this ABI into
the real `src/` value model, and B2 links the same primitive/runtime library
into compiled output.
