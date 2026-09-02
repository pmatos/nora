#include <catch2/catch.hpp>

#include "nora_rt.h"

#include <gc.h>

// S0 of the value-model + GC migration: the Boehm collector is linked and
// initialised, and the reusable nr_value immediate ABI is usable. This test
// touches only immediates and the collector — never a polymorphic-cell object
// accessor (M2 cells have a vptr at offset 0, not an ObjHeader).
TEST_CASE("libgc links, inits, and the nr_value immediate ABI is usable",
          "[m2][gc]") {
  REQUIRE(nr_fixnum_val(nr_fixnum(42)) == 42);
  REQUIRE(nr_fixnum_val(nr_fixnum(-7)) == -7);
  REQUIRE(nr_truthy(nr_bool(true)));
  REQUIRE_FALSE(nr_truthy(NR_FALSE));
  REQUIRE(nr_char_val(nr_char('Z')) == 'Z');

  void *P = GC_MALLOC(64);
  REQUIRE(P != nullptr);
  REQUIRE(GC_get_heap_size() > 0);
  REQUIRE(nrt_gc_heap_size() > 0);
}
