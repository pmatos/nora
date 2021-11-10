#pragma once

#include <cstdint>
#include <gmp.h>
#include <string>

// Defines the AST of Linklet programs
// Access to the nodes is done using the visitor pattern.
namespace nir {

class Integer {
public:
  explicit Integer(const std::string &V);
  explicit Integer(int64_t V);
  Integer(const Integer &Int) = default;
  Integer(Integer &&Int) = default;
  Integer &operator=(const Integer &Int);
  Integer &operator=(Integer &&Int) = default;

  Integer &operator+=(const Integer &Int);

  bool operator==(int64_t X) const;
  bool operator==(const Integer &Int) const;

  void dump() const;
  ~Integer() = default;

private:
  mpz_t Value;
};

}; // namespace nir