# Spike R1 — value model / GC ABI

Throwaway proof-of-concept for issue #88. **Not part of the `norac` build.** It
freezes the shared runtime value representation that milestone M2 (interpreter)
and B0/B2 (compiler + `libnora_rt`) both consume. The design it validates is
documented in [`docs/value-model-abi.md`](../../docs/value-model-abi.md).

## What it proves

1. **One representation, two drivers.** A tiny tree-walking interpreter
   (`interp_loop`) and a statically compiled C++ function (`compiled_loop`,
   standing in for codegen output) compute the identical result while calling
   the *same* runtime entry points (`nrt.h`) on the *same* heap.
2. **Flat closures work in both.** A closure capturing a value is applied via
   the same `nrt_apply` from the interpreter and from compiled code.
3. **Boehm GC keeps a garbage loop bounded.** ~2 allocations/iteration for tens
   of millions of iterations churn multiple GiB through a sub-MiB live heap.

## Run

```
make          # build with pkg-config bdw-gc
./spike       # defaults: 5,000,000 interp iters; 50,000,000 GC-pressure iters
./spike 2000000 20000000
make asan     # AddressSanitizer build (LSan off; the GC heap is intentionally "leaked")
```

Expected tail: `ALL PROOFS PASSED`.

## Files

- `nrt.h` — the frozen ABI (tagged `nr_value`, `ObjHeader`, flat `NrClosure`,
  `nr_code` signature) + runtime API.
- `nrt.cpp` — runtime over Boehm GC (constructors, `eq?`, interned symbols,
  fixnum arithmetic, flat-closure apply).
- `spike.cpp` — mini interpreter, compiled-equivalent, and the two proofs.
