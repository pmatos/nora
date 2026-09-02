// NORA runtime — the immediate (tagged nr_value) ABI layer.
//
// Promoted from the R1 spike (spike/r1-value-model) and frozen in
// docs/value-model-abi.md. M2 reuses ONLY this immediate/tag layer over the
// existing polymorphic ValueNode hierarchy: heap values are GC-allocated C++
// cells whose vptr sits at offset 0 — NOT an ObjHeader. R1's ObjHeader-based
// object accessors (NrPair/NrBox/nrt_cons/…) are deliberately absent here so
// nothing reads a vtable pointer as a type tag; flattening the object layout to
// ObjHeader/nr_value is a post-M2 step. See docs/value-model-gc-migration.md.
#ifndef NORA_RT_H
#define NORA_RT_H

#include <cstddef>
#include <cstdint>

// A tagged 64-bit word:
//   bit 0 == 1                   fixnum   (value = (int64)w >> 1; 63-bit)
//   low 3 bits == 0b000 (w != 0) heap pointer (an 8-byte-aligned cell)
//   low 3 bits == 0b010          singleton immediate (subtype in w >> 3)
//   low 3 bits == 0b110          character (codepoint in w >> 3)
typedef uint64_t nr_value;

static constexpr uint64_t NR_TAG_MASK = 0x7;
static constexpr uint64_t NR_TAG_PTR = 0x0;
static constexpr uint64_t NR_TAG_FIX = 0x1;
static constexpr uint64_t NR_TAG_IMM = 0x2;
static constexpr uint64_t NR_TAG_CHR = 0x6;

enum NrImm : uint64_t {
  NR_IMM_FALSE = 0,
  NR_IMM_TRUE = 1,
  NR_IMM_NULL = 2,
  NR_IMM_VOID = 3,
  NR_IMM_EOF = 4,
  NR_IMM_UNDEF = 5,
  NR_IMM_UNINIT = 6,
};

#define NR_MK_IMM(sub) (((uint64_t)(sub) << 3) | NR_TAG_IMM)
static constexpr nr_value NR_FALSE = NR_MK_IMM(NR_IMM_FALSE);
static constexpr nr_value NR_TRUE = NR_MK_IMM(NR_IMM_TRUE);
static constexpr nr_value NR_NULL = NR_MK_IMM(NR_IMM_NULL);
static constexpr nr_value NR_VOID = NR_MK_IMM(NR_IMM_VOID);
static constexpr nr_value NR_EOF = NR_MK_IMM(NR_IMM_EOF);
static constexpr nr_value NR_UNDEF = NR_MK_IMM(NR_IMM_UNDEF);
static constexpr nr_value NR_UNINIT = NR_MK_IMM(NR_IMM_UNINIT);

static inline bool nr_is_fixnum(nr_value w) { return (w & NR_TAG_FIX) != 0; }
static inline bool nr_is_ptr(nr_value w) {
  return w != 0 && (w & NR_TAG_MASK) == NR_TAG_PTR;
}
static inline bool nr_is_imm(nr_value w) {
  return (w & NR_TAG_MASK) == NR_TAG_IMM;
}
static inline bool nr_is_char(nr_value w) {
  return (w & NR_TAG_MASK) == NR_TAG_CHR;
}

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

// Boehm GC heap statistics — the M2 forcing seam (a depth-independent live-heap
// plateau against unbounded churn). Declared without pulling <gc.h> into every
// includer; implemented in nora_rt.cpp.
size_t nrt_gc_heap_size(void);
size_t nrt_gc_total_bytes(void);

#endif // NORA_RT_H
