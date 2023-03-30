#pragma once

namespace nir {

class BooleanLiteral {
public:
  BooleanLiteral() = delete;
  BooleanLiteral(bool V) : Value(V) {}
  BooleanLiteral(const BooleanLiteral &) = default;
  BooleanLiteral(BooleanLiteral &&) = default;
  BooleanLiteral &operator=(const BooleanLiteral &) = default;
  BooleanLiteral &operator=(BooleanLiteral &&) = default;
  ~BooleanLiteral() = default;

  [[nodiscard]] bool value() const { return Value; }

private:
  bool Value;
};

}; // namespace nir