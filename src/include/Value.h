#ifndef NORA_VALUE_H
#define NORA_VALUE_H

#include <concepts>
#include <memory>
#include <utility>

#include "AST.h"

// A machine value handle — the vehicle for the value-model + GC migration
// (docs/value-model-gc-migration.md §3). It will become a bare nr_value word
// (immediate | GC pointer | legacy pin-index) so GC cells can hold it. In the
// current phase it carries either an exclusively-owned legacy ValueNode
// (behaviourally identical to a plain unique_ptr) or a shared reference into
// an Environment binding (share()), so a lookup can hand out a value without
// cloning it. At most one of the two alternatives is engaged at a time.
class Value {
public:
  Value() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): implicit, for `reg = nullptr`.
  Value(std::nullptr_t) {}
  // NOLINTNEXTLINE(google-explicit-constructor): implicit boundary from legacy.
  template <std::derived_from<ast::ValueNode> T>
  Value(std::unique_ptr<T> V) : Legacy(std::move(V)) {}
  Value(Value &&) = default;
  Value &operator=(Value &&) = default;
  Value(const Value &) = delete;
  Value &operator=(const Value &) = delete;
  ~Value() = default;

  // Wrap an environment-shared reference. Two Values built from the same
  // shared_ptr alias the same underlying ValueNode.
  static Value share(std::shared_ptr<ast::ValueNode> V) {
    Value Result;
    Result.Shared = std::move(V);
    return Result;
  }

  // Get a shared_ptr to the held value, for storing into an Environment
  // binding. Moves Legacy into a fresh shared_ptr (if engaged), or returns a
  // copy of Shared (if engaged).
  std::shared_ptr<ast::ValueNode> toShared() {
    if (Legacy) {
      return std::shared_ptr<ast::ValueNode>(std::move(Legacy));
    }
    return Shared;
  }

  explicit operator bool() const {
    return static_cast<bool>(Legacy) || static_cast<bool>(Shared);
  }
  ast::ValueNode *get() const { return Legacy ? Legacy.get() : Shared.get(); }
  // Move the value out as an exclusively-owned legacy pointer, emptying this
  // handle. If Legacy is engaged this is a plain move (no extra cost). If
  // Shared is engaged, the caller needs exclusive ownership (e.g. to store
  // into a still-unique_ptr-typed Frame slot), so materialize a private copy.
  std::unique_ptr<ast::ValueNode> takeLegacy() {
    if (Legacy) {
      return std::move(Legacy);
    }
    if (Shared) {
      auto Owned = std::unique_ptr<ast::ValueNode>(Shared->clone());
      Shared.reset();
      return Owned;
    }
    return nullptr;
  }

private:
  std::unique_ptr<ast::ValueNode> Legacy;
  std::shared_ptr<ast::ValueNode> Shared;
};

#endif // NORA_VALUE_H
