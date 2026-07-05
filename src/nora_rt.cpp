#include "nora_rt.h"

#include <gc.h>

size_t nrt_gc_heap_size(void) { return GC_get_heap_size(); }
size_t nrt_gc_total_bytes(void) { return GC_get_total_bytes(); }
