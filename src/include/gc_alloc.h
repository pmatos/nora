#ifndef NORA_GC_ALLOC_H
#define NORA_GC_ALLOC_H

#include <cstddef>

#include <gc.h>

// A minimal, exception-free C++ allocator over the Boehm-Demers-Weiser GC.
//
// It allocates *scanned* memory (GC_MALLOC), so any pointers stored in the
// container's backing buffer are traced by the collector — the point of moving
// a container onto the GC heap in the value-model migration. Boehm's own
// gc_allocator is unusable here because it requires -fexceptions, which this
// codebase disables (LLVM default); this is the exception-free equivalent.
//
// deallocate() is intentionally a no-op: the collector reclaims a buffer once
// nothing references it, which is the conservative-GC-safe choice (an explicit
// GC_FREE could free a buffer a stale conservative pointer still appears to
// reference). Element construction/destruction still happen normally via
// std::allocator_traits, so RAII members of legacy elements are not leaked.
template <class T> struct GcAllocator {
  using value_type = T;

  GcAllocator() = default;
  template <class U> GcAllocator(const GcAllocator<U> &) noexcept {}

  T *allocate(std::size_t N) {
    return static_cast<T *>(GC_MALLOC(N * sizeof(T)));
  }
  void deallocate(T *, std::size_t) noexcept {}

  template <class U> bool operator==(const GcAllocator<U> &) const noexcept {
    return true;
  }
  template <class U> bool operator!=(const GcAllocator<U> &) const noexcept {
    return false;
  }
};

#endif // NORA_GC_ALLOC_H
