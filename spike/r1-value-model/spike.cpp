// R1 spike driver. Proves two things about the shared nr_value ABI:
//
//  (1) EQUIVALENCE: a tree-walking interpreter and a statically compiled C++
//      function (standing in for codegen output) compute identical results
//      while calling the *same* runtime entry points on the *same* heap.
//  (2) GC: a garbage-generating tail loop runs in bounded heap — Boehm collects
//      the per-iteration boxes/pairs, so cumulative bytes allocated dwarf the
//      live heap.
//
// Build: make (see Makefile). Run: ./spike [N_interp] [N_gc]
#include "nrt.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <gc.h>
#include <sys/resource.h>

// ---------------------------------------------------------------------------
// A tiny AST + trampolining tree-walker. Enough to express a tail-recursive
// loop that allocates (and drops) a box and a pair each iteration.
// ---------------------------------------------------------------------------
enum EK {
  LIT,
  CLOSLIT,
  VAR,
  IFE,
  ADD,
  SUB,
  BOXV,
  UNBOXV,
  CONSV,
  CARV,
  EQ0,
  RECUR,
  CALLV,
};

struct Expr {
  EK k;
  nr_value lit = 0;               // LIT / CLOSLIT
  int var = 0;                    // VAR: env index
  const Expr *a = nullptr;        // first child
  const Expr *b = nullptr;        // second child
  const Expr *c = nullptr;        // IFE else-branch
  std::vector<const Expr *> args; // RECUR / CALLV
  explicit Expr(EK kk) : k(kk) {} // ctor -> no aggregate-init warnings
};

struct RecurSig {
  bool active = false;
  std::vector<nr_value> next;
};

static nr_value eval(const Expr *e, const std::vector<nr_value> &env,
                     RecurSig &rec);

// Non-tail evaluation: recur must not escape a non-tail position.
static nr_value evalNT(const Expr *e, const std::vector<nr_value> &env) {
  RecurSig local;
  nr_value v = eval(e, env, local);
  assert(!local.active && "recur in non-tail position");
  return v;
}

static nr_value eval(const Expr *e, const std::vector<nr_value> &env,
                     RecurSig &rec) {
  switch (e->k) {
  case LIT:
  case CLOSLIT:
    return e->lit;
  case VAR:
    return env[(size_t)e->var];
  case ADD:
    return nrt_fx_add(evalNT(e->a, env), evalNT(e->b, env));
  case SUB:
    return nrt_fx_sub(evalNT(e->a, env), evalNT(e->b, env));
  case BOXV:
    return nrt_box(evalNT(e->a, env));
  case UNBOXV:
    return nrt_unbox(evalNT(e->a, env));
  case CONSV:
    return nrt_cons(evalNT(e->a, env), evalNT(e->b, env));
  case CARV:
    return nrt_car(evalNT(e->a, env));
  case EQ0:
    return nr_bool(nrt_eq(evalNT(e->a, env), nr_fixnum(0)));
  case IFE:
    // condition is non-tail; the chosen branch is tail (recur may propagate).
    return nr_truthy(evalNT(e->a, env)) ? eval(e->b, env, rec)
                                        : eval(e->c, env, rec);
  case CALLV: {
    nr_value f = evalNT(e->a, env);
    std::vector<nr_value> as;
    as.reserve(e->args.size());
    for (auto *ae : e->args)
      as.push_back(evalNT(ae, env));
    return nrt_apply(f, (int64_t)as.size(), as.data());
  }
  case RECUR: {
    rec.next.clear();
    rec.next.reserve(e->args.size());
    for (auto *ae : e->args)
      rec.next.push_back(evalNT(ae, env));
    rec.active = true;
    return NR_VOID;
  }
  }
  abort();
}

static nr_value run_loop(const Expr *body, std::vector<nr_value> env) {
  for (;;) {
    RecurSig rec;
    nr_value v = eval(body, env, rec);
    if (!rec.active)
      return v;
    env = std::move(rec.next);
  }
}

// ---------------------------------------------------------------------------
// The interpreted loop:  (rec loop ([n N] [sum 0])
//   (if (= n 0) sum (loop (- n 1) (+ (unbox (box n)) (car (cons sum sum))))))
// sum' = sum + n, with a box and a pair allocated (and dropped) per iteration.
// ---------------------------------------------------------------------------
static nr_value interp_loop(int64_t N) {
  Expr n{VAR};
  n.var = 0;
  Expr sum{VAR};
  sum.var = 1;
  Expr zero{EQ0};
  zero.a = &n;

  Expr one{LIT};
  one.lit = nr_fixnum(1);
  Expr nm1{SUB};
  nm1.a = &n;
  nm1.b = &one;

  Expr boxn{BOXV};
  boxn.a = &n; // (box n)  -> garbage
  Expr ubox{UNBOXV};
  ubox.a = &boxn; // (unbox (box n)) = n
  Expr consss{CONSV};
  consss.a = &sum;
  consss.b = &sum; // (cons sum sum) -> garbage
  Expr carp{CARV};
  carp.a = &consss; // (car (cons sum sum)) = sum
  Expr sump{ADD};
  sump.a = &ubox;
  sump.b = &carp; // n + sum

  Expr recur{RECUR};
  recur.args = {&nm1, &sump};
  Expr body{IFE};
  body.a = &zero;
  body.b = &sum;
  body.c = &recur;

  return run_loop(&body, {nr_fixnum(N), nr_fixnum(0)});
}

// The compiled-equivalent loop: identical runtime calls, no interpreter.
static nr_value compiled_loop(int64_t N) {
  nr_value n = nr_fixnum(N), sum = nr_fixnum(0);
  while (!nrt_eq(n, nr_fixnum(0))) {
    nr_value tmp = nrt_unbox(nrt_box(n));     // box garbage
    nr_value s = nrt_car(nrt_cons(sum, sum)); // pair garbage
    sum = nrt_fx_add(tmp, s);
    n = nrt_fx_sub(n, nr_fixnum(1));
  }
  return sum;
}

// ---------------------------------------------------------------------------
// Flat-closure proof: a closure capturing one value, applied in both paths.
// ---------------------------------------------------------------------------
static nr_value add_captured(nr_value self, int64_t argc,
                             const nr_value *argv) {
  assert(argc == 1);
  auto *c = (NrClosure *)self;
  return nrt_fx_add(c->free[0], argv[0]);
}

static long rss_kb() {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss; // KiB on Linux
}

static void hr(const char *label, size_t bytes) {
  printf("  %-22s %10.2f MiB\n", label, (double)bytes / (1024 * 1024));
}

int main(int argc, char **argv) {
  nrt_init();
  int64_t N = argc > 1 ? atoll(argv[1]) : 5000000;    // interpreter loop
  int64_t Ngc = argc > 2 ? atoll(argv[2]) : 50000000; // GC-pressure loop
  int fails = 0;

  printf("== R1 value-model / GC spike ==\n");
  printf(
      "nr_value = %zu bytes;  ObjHeader = %zu;  NrPair = %zu;  NrBox = %zu;  "
      "NrClosure = %zu\n\n",
      sizeof(nr_value), sizeof(ObjHeader), sizeof(NrPair), sizeof(NrBox),
      sizeof(NrClosure));

  // --- immediate/tag sanity -------------------------------------------------
  assert(nr_is_fixnum(nr_fixnum(-42)) && nr_fixnum_val(nr_fixnum(-42)) == -42);
  assert(nr_is_fixnum(nr_fixnum(0)) && nr_fixnum_val(nr_fixnum(0)) == 0);
  assert(nr_is_char(nr_char('Z')) && nr_char_val(nr_char('Z')) == 'Z');
  assert(!nr_truthy(NR_FALSE) && nr_truthy(NR_TRUE) && nr_truthy(nr_fixnum(0)));
  assert(nr_is_ptr(nrt_cons(nr_fixnum(1), NR_NULL)));
  // identity: interned symbols eq?, fresh pairs not eq?.
  assert(nrt_eq(nrt_intern("lambda"), nrt_intern("lambda")));
  assert(!nrt_eq(nrt_intern("lambda"), nrt_intern("if")));
  nr_value p = nrt_cons(nr_fixnum(1), nr_fixnum(2));
  assert(nrt_eq(p, p) && !nrt_eq(p, nrt_cons(nr_fixnum(1), nr_fixnum(2))));
  // mutation through a shared reference (no clone!).
  nr_value b = nrt_box(nr_fixnum(7));
  nr_value alias = b;
  nrt_set_box(alias, nr_fixnum(99));
  assert(nr_fixnum_val(nrt_unbox(b)) == 99);
  printf("[ok] tagging, identity, interning, in-place mutation\n");

  // --- PROOF 1: interpreter == compiled ------------------------------------
  int64_t expected = N % 2 == 0 ? (N / 2) * (N + 1) : N * ((N + 1) / 2);
  nr_value ri = interp_loop(N);
  nr_value rc = compiled_loop(N);
  printf("[%.4s] interp(%lld) = ", nrt_eq(ri, rc) ? "ok" : "FAIL",
         (long long)N);
  nrt_write(ri);
  printf(" ; compiled = ");
  nrt_write(rc);
  printf(" ; expected N(N+1)/2 = %lld\n", (long long)expected);
  if (!nrt_eq(ri, rc) || nr_fixnum_val(ri) != expected)
    fails++;

  // flat closure applied through both paths
  nr_value cap = nr_fixnum(10);
  nr_value clos = nrt_make_closure(add_captured, 1, &cap);
  nr_value five = nr_fixnum(5);
  nr_value r_compiled = nrt_apply(clos, 1, &five);
  Expr cl{CLOSLIT};
  cl.lit = clos;
  Expr a5{LIT};
  a5.lit = five;
  Expr call{CALLV};
  call.a = &cl;
  call.args = {&a5};
  nr_value r_interp = evalNT(&call, {});
  printf(
      "[%.4s] closure (capture 10) applied to 5: interp=%lld compiled=%lld\n",
      (nrt_eq(r_interp, r_compiled) && nr_fixnum_val(r_interp) == 15) ? "ok"
                                                                      : "FAIL",
      (long long)nr_fixnum_val(r_interp), (long long)nr_fixnum_val(r_compiled));
  if (!nrt_eq(r_interp, r_compiled) || nr_fixnum_val(r_interp) != 15)
    fails++;

  // --- PROOF 2: GC keeps a garbage loop bounded ----------------------------
  size_t heap_before = nrt_gc_heap_size();
  size_t total_before = nrt_gc_total_bytes();
  nr_value g = compiled_loop(Ngc); // ~2 allocations per iteration
  (void)g;
  size_t heap_after = nrt_gc_heap_size();
  size_t total_after = nrt_gc_total_bytes();
  size_t churn = total_after - total_before;
  printf("\nGC pressure loop: %lld iterations (~2 allocs each)\n",
         (long long)Ngc);
  hr("cumulative allocated", churn);
  hr("live heap after loop", heap_after);
  hr("heap before loop", heap_before);
  printf("  %-22s %10ld MiB (getrusage peak)\n", "process RSS",
         rss_kb() / 1024);
  bool collected = churn > (size_t)1 << 30 && // > 1 GiB churned
                   heap_after < churn / 10;   // live heap < 10% of churn
  printf("[%.4s] GC collected: churned %.2f GiB into a %.1f MiB live heap "
         "(%.0fx)\n",
         collected ? "ok" : "FAIL", (double)churn / (1 << 30),
         (double)heap_after / (1024 * 1024),
         (double)churn / (double)heap_after);
  if (!collected)
    fails++;
  (void)total_before;
  (void)heap_before;

  printf("\n%s\n", fails == 0 ? "ALL PROOFS PASSED" : "SPIKE FAILED");
  return fails == 0 ? 0 : 1;
}
