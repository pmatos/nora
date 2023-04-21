#pragma once

#include <llvm/Support/Casting.h>
#include <memory>
#include <type_traits>

// Implements a wrapper for llvm::dyn_cast to use with smart_pointers.
template <typename To, typename From>
std::unique_ptr<To> dyn_castU(std::unique_ptr<From> &Ptr) {
  static_assert(std::is_base_of_v<From, To> || std::is_base_of_v<To, From>,
                "To and From must be in the same inheritance hierarchy");

  // If Ptr is null, we just return it
  if (!Ptr) {
    return std::unique_ptr<To>();
  }

  // Perform dynamic_cast using raw pointers
  To *CastedRawPtr = llvm::dyn_cast<To>(Ptr.get());

  if (CastedRawPtr) {
    // Release ownership from the original unique_ptr
    Ptr.release();

    // Return a new unique_ptr of the target type
    return std::unique_ptr<To>(CastedRawPtr);
  }

  // Return an empty unique_ptr if the dynamic_cast failed
  return std::unique_ptr<To>();
}

template <typename To, typename From>
std::shared_ptr<To> dyn_castS(const std::shared_ptr<From> &Ptr) {
  static_assert(std::is_base_of_v<From, To> || std::is_base_of_v<To, From>,
                "To and From must be in the same inheritance hierarchy");

  // If Ptr is null, we just return it
  if (!Ptr) {
    return std::unique_ptr<To>();
  }

  // Perform dynamic_cast using raw pointers
  To *CastedRawPtr = llvm::dyn_cast<To *>(Ptr.get());

  if (CastedRawPtr) {
    // Return a new shared_ptr of the target type
    return std::shared_ptr<To>(Ptr, CastedRawPtr);
  }

  // Return an empty shared_ptr if the dynamic_cast failed
  return std::shared_ptr<To>();
}
