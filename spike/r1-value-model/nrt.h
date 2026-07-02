// R1 spike — NORA runtime value ABI (nr_value) + Boehm-GC heap.
//
// This is the *frozen* representation that milestone M2 (interpreter value
// model) and B0/B2 (NIR concrete types + libnora_rt) are both meant to consume,
// so that the tree-walking interpreter and statically compiled code share one
// heap and one object layout. See docs/value-model-abi.md for the rationale.
//
// Throwaway spike: standalone, not wired into the norac build.
#ifndef NRT_H
#define NRT_H

#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------------------
// nr_value: a tagged 64-bit word.
//
//   bit 0 == 1                      -> fixnum   (value = (int64)w >> 1; 63-bit)
//   low 3 bits == 0b000 (w != 0)    -> heap pointer (8-byte-aligned ObjHeader*)
//   low 3 bits == 0b010             -> singleton immediate (subtype in w >> 3)
//   low 3 bits == 0b110             -> character (codepoint in w >> 3)
//   low 3 bits == 0b100             -> reserved
//
// Fixnums are odd, so they never collide with the even-tagged immediates or the
// 8-aligned heap pointers. Heap pointers keep tag 000 so an ObjHeader* is its
// own nr_value with no masking on dereference.
// ---------------------------------------------------------------------------
typedef uint64_t nr_value;

static constexpr uint64_t NR_TAG_MASK = 0x7;
static constexpr uint64_t NR_TAG_PTR = 0x0;  // heap pointer
static constexpr uint64_t NR_TAG_FIX = 0x1;  // fixnum (any odd word)
static constexpr uint64_t NR_TAG_IMM = 0x2;  // singleton immediate
static constexpr uint64_t NR_TAG_CHR = 0x6;  // character

// Singleton immediates: subtype << 3 | NR_TAG_IMM.
enum NrImm : uint64_t {
  NR_IMM_FALSE = 0,
  NR_IMM_TRUE = 1,
  NR_IMM_NULL = 2,   // '()
  NR_IMM_VOID = 3,   // (void)
  NR_IMM_EOF = 4,
  NR_IMM_UNDEF = 5,  // unsafe-undefined sentinel (729x in the expander)
  NR_IMM_UNINIT = 6, // letrec pre-initialisation hole
};

#define NR_MK_IMM(sub) (((uint64_t)(sub) << 3) | NR_TAG_IMM)
static constexpr nr_value NR_FALSE = NR_MK_IMM(NR_IMM_FALSE);
static constexpr nr_value NR_TRUE = NR_MK_IMM(NR_IMM_TRUE);
static constexpr nr_value NR_NULL = NR_MK_IMM(NR_IMM_NULL);
static constexpr nr_value NR_VOID = NR_MK_IMM(NR_IMM_VOID);
static constexpr nr_value NR_EOF = NR_MK_IMM(NR_IMM_EOF);
static constexpr nr_value NR_UNDEF = NR_MK_IMM(NR_IMM_UNDEF);
static constexpr nr_value NR_UNINIT = NR_MK_IMM(NR_IMM_UNINIT);

// --- immediate predicates / (un)boxing -------------------------------------
static inline bool nr_is_fixnum(nr_value w) { return (w & NR_TAG_FIX) != 0; }
static inline bool nr_is_ptr(nr_value w) {
  return w != 0 && (w & NR_TAG_MASK) == NR_TAG_PTR;
}
static inline bool nr_is_imm(nr_value w) { return (w & NR_TAG_MASK) == NR_TAG_IMM; }
static inline bool nr_is_char(nr_value w) { return (w & NR_TAG_MASK) == NR_TAG_CHR; }

static inline nr_value nr_fixnum(int64_t v) {
  return (nr_value)((uint64_t)v << 1) | NR_TAG_FIX;
}
static inline int64_t nr_fixnum_val(nr_value w) {
  return (int64_t)w >> 1; // arithmetic shift keeps the sign
}
static inline nr_value nr_char(uint32_t cp) {
  return ((nr_value)cp << 3) | NR_TAG_CHR;
}
static inline uint32_t nr_char_val(nr_value w) { return (uint32_t)(w >> 3); }
static inline nr_value nr_bool(bool b) { return b ? NR_TRUE : NR_FALSE; }

// Racket truthiness: only #f is false.
static inline bool nr_truthy(nr_value w) { return w != NR_FALSE; }

// ---------------------------------------------------------------------------
// Heap objects. Every heap object begins with an 8-byte ObjHeader so payloads
// stay 8-aligned. `meta` carries a length / arity / flags per type. (A precise
// GC would also live here; the spike uses Boehm conservative GC and needs only
// the type tag — see the ABI doc.)
// ---------------------------------------------------------------------------
enum NrObjType : uint32_t {
  OBJ_PAIR = 1,
  OBJ_BOX,
  OBJ_CLOSURE,
  OBJ_SYMBOL,
};

struct ObjHeader {
  uint32_t type; // NrObjType
  uint32_t meta; // length / arity / flags
};

struct NrPair {
  ObjHeader h;
  nr_value car;
  nr_value cdr;
};

struct NrBox {
  ObjHeader h;
  nr_value val;
};

// A flat closure: header + code pointer + inline captured cells. This layout is
// byte-for-byte what B2's compiled closures use, and `nr_code` is exactly B2's
// planned code signature (self is the closure; it reads free[] for captures).
typedef nr_value (*nr_code)(nr_value self, int64_t argc, const nr_value *argv);

struct NrClosure {
  ObjHeader h;
  nr_code code;
  uint32_t nfree;
  uint32_t pad;
  nr_value free[]; // flexible array member, 8-aligned
};

struct NrSymbol {
  ObjHeader h;
  const char *name;
  uint64_t hash;
};

static inline ObjHeader *nr_obj(nr_value w) { return (ObjHeader *)w; }
static inline bool nr_has_type(nr_value w, NrObjType t) {
  return nr_is_ptr(w) && nr_obj(w)->type == t;
}

// ---------------------------------------------------------------------------
// Runtime API (implemented in nrt.cpp). Deliberately identical entry points for
// the interpreter and for compiled code.
// ---------------------------------------------------------------------------
void nrt_init(void); // GC_INIT + intern table

// pairs
nr_value nrt_cons(nr_value a, nr_value d);
nr_value nrt_car(nr_value p);
nr_value nrt_cdr(nr_value p);

// boxes
nr_value nrt_box(nr_value v);
nr_value nrt_unbox(nr_value b);
void nrt_set_box(nr_value b, nr_value v);

// symbols (interned; eq? works by pointer identity)
nr_value nrt_intern(const char *name);

// closures
nr_value nrt_make_closure(nr_code code, uint32_t nfree, const nr_value *freev);
nr_value nrt_apply(nr_value clos, int64_t argc, const nr_value *argv);

// identity / equality
static inline bool nrt_eq(nr_value a, nr_value b) { return a == b; } // eq?

// fixnum arithmetic (spike: no bignum promotion; asserts on overflow)
nr_value nrt_fx_add(nr_value a, nr_value b);
nr_value nrt_fx_sub(nr_value a, nr_value b);

// debug
void nrt_write(nr_value w);

// GC statistics passthrough (Boehm)
size_t nrt_gc_heap_size(void);
size_t nrt_gc_total_bytes(void);

#endif // NRT_H
