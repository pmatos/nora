#ifndef NORA_VALUE_H
#define NORA_VALUE_H

#include <memory>

#include "AST.h"

// A machine value handle — the vehicle for the value-model + GC migration
// (docs/value-model-gc-migration.md §3). It will become a bare nr_value word
// (immediate | GC pointer | legacy pin-index) so GC cells can hold it; in the
// current phase it simply carries a legacy heap ValueNode by unique_ptr and is
// behaviourally identical to it. Move-only, like the unique_ptr it wraps.
class Value {
public:
  Value() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): implicit, for `reg = nullptr`.
  Value(std::nullptr_t) {}
  // NOLINTNEXTLINE(google-explicit-constructor): implicit boundary from legacy.
  Value(std::unique_ptr<ast::ValueNode> V) : Legacy(std::move(V)) {}
  Value(Value &&) = default;
  Value &operator=(Value &&) = default;
  Value(const Value &) = delete;
  Value &operator=(const Value &) = delete;
  ~Value() = default;

  explicit operator bool() const { return static_cast<bool>(Legacy); }
  ast::ValueNode *get() const { return Legacy.get(); }
  // Move the legacy value out, emptying this handle. Used at boundaries with
  // the still-unique_ptr frame/env slots until they migrate in later slices.
  std::unique_ptr<ast::ValueNode> takeLegacy() { return std::move(Legacy); }

private:
  std::unique_ptr<ast::ValueNode> Legacy;
};

#endif // NORA_VALUE_H
