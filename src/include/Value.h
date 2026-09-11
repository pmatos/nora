#ifndef NORA_VALUE_H
#define NORA_VALUE_H

#include <cassert>
#include <concepts>
#include <memory>
#include <utility>

#include <llvm/Support/Casting.h>

#include "AST.h"
#include "nora_rt.h"

// A machine value handle — the vehicle for the value-model + GC migration
// (docs/value-model-gc-migration.md §3). It will become a bare nr_value word
// (immediate | GC pointer | legacy pin-index) so GC cells can hold it. In the
// current phase it carries an exclusively-owned legacy ValueNode (behaviourally
// identical to a plain unique_ptr), a shared reference into an Environment
// binding (share()), so a lookup can hand out a value without cloning it, or a
// bare nr_value immediate word (immediate(), M2/GC S6+) for allocation-free
// values such as booleans. At most one of the three alternatives is engaged at
// a time; Legacy and Imm are mutable so a const accessor (get()) can still
// materialize an immediate into Legacy on first use.
class Value {
public:
  Value() = default;
  // NOLINTNEXTLINE(google-explicit-constructor): implicit, for `reg = nullptr`.
  Value(std::nullptr_t) {}
  // NOLINTNEXTLINE(google-explicit-constructor): implicit boundary from legacy.
  template <std::derived_from<ast::ValueNode> T>
  Value(std::unique_ptr<T> V) : Legacy(std::move(V)) {}
  // Hand-written (not `= default`): a scalar member's implicit move is a copy,
  // and moving-out must empty Imm the same way it empties Legacy/Shared.
  Value(Value &&Other) noexcept
      : Legacy(std::move(Other.Legacy)), Shared(std::move(Other.Shared)),
        Imm(std::exchange(Other.Imm, 0)) {}
  Value &operator=(Value &&Other) noexcept {
    Legacy = std::move(Other.Legacy);
    Shared = std::move(Other.Shared);
    Imm = std::exchange(Other.Imm, 0);
    return *this;
  }
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

  // Wrap a bare nr_value immediate word. W must not be 0 (0 is "unengaged"
  // under every tag: fixnums set bit 0, heap pointers/immediates/chars are all
  // nonzero by construction).
  static Value immediate(nr_value W) {
    // NOLINTNEXTLINE(misc-static-assert): a runtime check, not a constant.
    assert(W != 0);
    Value Result;
    Result.Imm = W;
    return Result;
  }

  // Build a legacy ValueNode view of an engaged immediate word, so every
  // existing consumer that expects a non-null ast::ValueNode* keeps working
  // unchanged. The single dispatcher shared by materializeLegacy() and
  // Interpreter::getResult(), so a later slice adding another immediate kind
  // (fixnums) extends this one place rather than re-deriving a second copy.
  static std::unique_ptr<ast::ValueNode> viewOf(nr_value W) {
    if (nr_is_char(W)) {
      return std::make_unique<ast::Char>(nr_char_val(W));
    }
    if (W == NR_VOID) {
      return std::make_unique<ast::Void>();
    }
    // NOLINTNEXTLINE(misc-static-assert): a runtime check, not a constant.
    assert(W == NR_TRUE || W == NR_FALSE);
    return std::make_unique<ast::BooleanLiteral>(nr_truthy(W));
  }

  // Move a shared_ptr to the held value out, for storing into an Environment
  // binding. Moves Legacy into a fresh shared_ptr (if engaged, materializing
  // an immediate into Legacy first), or moves Shared out directly (if
  // engaged). Either way, emptying this handle: unlike get(), this is a
  // one-shot consuming read, not a peek.
  [[nodiscard]] std::shared_ptr<ast::ValueNode> toShared() {
    materializeLegacy();
    if (Legacy) {
      return std::shared_ptr<ast::ValueNode>(std::move(Legacy));
    }
    return std::exchange(Shared, nullptr);
  }

  explicit operator bool() const {
    return static_cast<bool>(Legacy) || static_cast<bool>(Shared) || Imm != 0;
  }
  // Whether the immediate alternative is engaged (before any materialization
  // on this handle). Does not itself materialize.
  bool isImmediate() const { return Imm != 0; }
  // The raw immediate word. isImmediate() must hold.
  nr_value rawImmediate() const {
    // NOLINTNEXTLINE(misc-static-assert): a runtime check, not a constant.
    assert(isImmediate());
    return Imm;
  }
  ast::ValueNode *get() const {
    materializeLegacy();
    return Legacy ? Legacy.get() : Shared.get();
  }
  // Borrow the held value as a specific ValueNode subtype, without consuming or
  // cloning it. Like get(), a non-consuming peek: a Shared handle hands back
  // the very node its binding shares (pointer identity preserved), and an
  // engaged immediate is materialized into Legacy exactly as get() does.
  // Returns nullptr if the handle is unengaged or the held node is not a T.
  // This is the typed peek used wherever an arm only inspects the register;
  // ownership transfer stays on the loud consuming paths
  // (takeLegacy()/toShared()).
  template <std::derived_from<ast::ValueNode> T>
  [[nodiscard]] const T *as() const {
    return llvm::dyn_cast_or_null<T>(get());
  }
  // Move the value out as an exclusively-owned legacy pointer, emptying this
  // handle. If Legacy is engaged (or an immediate just materialized into it)
  // this is a plain move (no extra cost). If Shared is engaged, the caller
  // needs exclusive ownership, so materialize a private copy — this is the only
  // path that clones. Reserved for splicing a value into a unique_ptr<ExprNode>
  // AST-child slot (ast::Values / ast::List); to hand a value to a sink that
  // already takes a Value (deliver/envSet/Environment::add/toShared) move the
  // whole handle, and to inspect it use get()/as<T>().
  [[nodiscard]] std::unique_ptr<ast::ValueNode> takeLegacy() {
    materializeLegacy();
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
  // Materialize an engaged immediate into Legacy via viewOf(), so every
  // existing consumer that expects a non-null ast::ValueNode* keeps working
  // unchanged.
  void materializeLegacy() const {
    if (Imm != 0 && !Legacy && !Shared) {
      Legacy = viewOf(Imm);
      Imm = 0;
    }
  }

  mutable std::unique_ptr<ast::ValueNode> Legacy;
  std::shared_ptr<ast::ValueNode> Shared;
  mutable nr_value Imm = 0;
};

#endif // NORA_VALUE_H
