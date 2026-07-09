// R1 spike — runtime implementation over the Boehm-Demers-Weiser collector.
#include "nrt.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#include <gc.h>

void nrt_init(void) {
  GC_INIT();
  // Deterministic behaviour for the spike; Boehm still collects on demand.
  GC_set_all_interior_pointers(1);
}

// --- allocation helpers ----------------------------------------------------
static void *alloc(size_t n) {
  void *p = GC_MALLOC(n); // zero-filled, scanned for pointers
  assert(p && "GC_MALLOC returned null");
  assert(((uintptr_t)p & NR_TAG_MASK) == 0 && "heap object not 8-aligned");
  return p;
}

// --- pairs -----------------------------------------------------------------
nr_value nrt_cons(nr_value a, nr_value d) {
  auto *p = (NrPair *)alloc(sizeof(NrPair));
  p->h = {OBJ_PAIR, 0};
  p->car = a;
  p->cdr = d;
  return (nr_value)p;
}
nr_value nrt_car(nr_value p) {
  assert(nr_has_type(p, OBJ_PAIR) && "car: not a pair");
  return ((NrPair *)p)->car;
}
nr_value nrt_cdr(nr_value p) {
  assert(nr_has_type(p, OBJ_PAIR) && "cdr: not a pair");
  return ((NrPair *)p)->cdr;
}

// --- boxes -----------------------------------------------------------------
nr_value nrt_box(nr_value v) {
  auto *b = (NrBox *)alloc(sizeof(NrBox));
  b->h = {OBJ_BOX, 0};
  b->val = v;
  return (nr_value)b;
}
nr_value nrt_unbox(nr_value b) {
  assert(nr_has_type(b, OBJ_BOX) && "unbox: not a box");
  return ((NrBox *)b)->val;
}
void nrt_set_box(nr_value b, nr_value v) {
  assert(nr_has_type(b, OBJ_BOX) && "set-box!: not a box");
  ((NrBox *)b)->val = v;
}

// --- symbols (interned; permanent, so uncollectable) -----------------------
static std::unordered_map<std::string, nr_value> &intern_table() {
  static std::unordered_map<std::string, nr_value> t;
  return t;
}
nr_value nrt_intern(const char *name) {
  auto &t = intern_table();
  std::string key(name);
  auto it = t.find(key);
  if (it != t.end())
    return it->second;
  // Interned symbols live forever: allocate uncollectable so the intern table
  // (a plain malloc'd container the GC does not scan) cannot dangle.
  auto *s = (NrSymbol *)GC_MALLOC_UNCOLLECTABLE(sizeof(NrSymbol));
  size_t n = key.size() + 1;
  char *buf = (char *)GC_MALLOC_ATOMIC_UNCOLLECTABLE(n);
  memcpy(buf, name, n);
  s->h = {OBJ_SYMBOL, 1};
  s->name = buf;
  s->hash = std::hash<std::string>{}(key);
  nr_value w = (nr_value)s;
  t.emplace(std::move(key), w);
  return w;
}

// --- closures (flat capture) ----------------------------------------------
nr_value nrt_make_closure(nr_code code, uint32_t nfree, const nr_value *freev) {
  auto *c =
      (NrClosure *)alloc(sizeof(NrClosure) + (size_t)nfree * sizeof(nr_value));
  c->h = {OBJ_CLOSURE, nfree};
  c->code = code;
  c->nfree = nfree;
  c->pad = 0;
  for (uint32_t i = 0; i < nfree; ++i)
    c->free[i] = freev[i];
  return (nr_value)c;
}
nr_value nrt_apply(nr_value clos, int64_t argc, const nr_value *argv) {
  assert(nr_has_type(clos, OBJ_CLOSURE) && "apply: not a closure");
  return ((NrClosure *)clos)->code(clos, argc, argv);
}

// --- fixnum arithmetic (spike: assert instead of promoting to bignum) ------
nr_value nrt_fx_add(nr_value a, nr_value b) {
  assert(nr_is_fixnum(a) && nr_is_fixnum(b));
  int64_t r;
  bool ovf = __builtin_add_overflow(nr_fixnum_val(a), nr_fixnum_val(b), &r);
  assert(!ovf && "fixnum add overflow (bignum promotion is M4, not R1)");
  (void)ovf;
  return nr_fixnum(r);
}
nr_value nrt_fx_sub(nr_value a, nr_value b) {
  assert(nr_is_fixnum(a) && nr_is_fixnum(b));
  int64_t r;
  bool ovf = __builtin_sub_overflow(nr_fixnum_val(a), nr_fixnum_val(b), &r);
  assert(!ovf && "fixnum sub overflow");
  (void)ovf;
  return nr_fixnum(r);
}

// --- debug printer ---------------------------------------------------------
void nrt_write(nr_value w) {
  if (nr_is_fixnum(w)) {
    printf("%lld", (long long)nr_fixnum_val(w));
  } else if (nr_is_char(w)) {
    printf("#\\%u", nr_char_val(w));
  } else if (w == NR_FALSE) {
    printf("#f");
  } else if (w == NR_TRUE) {
    printf("#t");
  } else if (w == NR_NULL) {
    printf("()");
  } else if (w == NR_VOID) {
    printf("#<void>");
  } else if (nr_has_type(w, OBJ_PAIR)) {
    printf("(");
    nrt_write(nrt_car(w));
    printf(" . ");
    nrt_write(nrt_cdr(w));
    printf(")");
  } else if (nr_has_type(w, OBJ_BOX)) {
    printf("#&");
    nrt_write(nrt_unbox(w));
  } else if (nr_has_type(w, OBJ_SYMBOL)) {
    printf("%s", ((NrSymbol *)w)->name);
  } else if (nr_has_type(w, OBJ_CLOSURE)) {
    printf("#<procedure>");
  } else {
    printf("#<0x%llx>", (unsigned long long)w);
  }
}

size_t nrt_gc_heap_size(void) { return GC_get_heap_size(); }
size_t nrt_gc_total_bytes(void) { return GC_get_total_bytes(); }
