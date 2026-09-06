#include <catch2/catch.hpp>

#include "AST.h"
#include "Value.h"

#include <llvm/Support/Casting.h>

#include <memory>
#include <utility>

// Slice 1 (issue #119): Value in isolation. No Environment involved yet.

TEST_CASE("Value constructs implicitly from a derived unique_ptr", "[value]") {
  Value V = std::make_unique<ast::Integer>(1);
  REQUIRE(V);
  auto *I = llvm::dyn_cast<ast::Integer>(V.get());
  REQUIRE(I);
  REQUIRE(*I == 1);
}

TEST_CASE("Value::share of the same shared_ptr yields shared identity",
          "[value]") {
  auto SP = std::make_shared<ast::Integer>(42);
  Value A = Value::share(SP);
  Value B = Value::share(SP);
  REQUIRE(A.get() == B.get());
  REQUIRE(A.get() == SP.get());
}

TEST_CASE("takeLegacy on a shared Value materializes a distinct-but-equal "
          "copy",
          "[value]") {
  auto SP = std::make_shared<ast::Integer>(7);
  Value V = Value::share(SP);
  std::unique_ptr<ast::ValueNode> Owned = V.takeLegacy();
  REQUIRE(Owned.get() != SP.get());
  auto *I = llvm::dyn_cast<ast::Integer>(Owned.get());
  REQUIRE(I);
  REQUIRE(*I == *SP);
}

TEST_CASE("takeLegacy on a plain unique_ptr-backed Value is a bare move",
          "[value]") {
  auto Original = std::make_unique<ast::Integer>(9);
  ast::Integer *RawPtr = Original.get();
  Value V = std::move(Original);
  std::unique_ptr<ast::ValueNode> Owned = V.takeLegacy();
  REQUIRE(Owned.get() == RawPtr);
}

TEST_CASE("toShared round-trips through a second share()", "[value]") {
  Value V = std::make_unique<ast::Integer>(3);
  std::shared_ptr<ast::ValueNode> SP1 = V.toShared();
  Value Shared1 = Value::share(SP1);
  Value Shared2 = Value::share(SP1);
  REQUIRE(Shared1.get() == Shared2.get());
  REQUIRE(Shared1.get() == SP1.get());
}
