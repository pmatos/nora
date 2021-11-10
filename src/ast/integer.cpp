#include "ast/integer.h"
#include <gmp.h>

using namespace nir;

Integer::Integer(const std::string &V) {
  mpz_init_set_str(Value, V.c_str(), 10);
}

Integer::Integer(int64_t V) { mpz_init_set_si(Value, V); }

Integer &Integer::operator=(const Integer &Int) {
  mpz_set(Value, Int.Value);
  return *this;
}

bool Integer::operator==(int64_t X) const { return mpz_cmp_si(Value, X) == 0; }
bool Integer::operator==(const Integer &Int) const {
  return mpz_cmp(Value, Int.Value) == 0;
}
Integer &Integer::operator+=(const Integer &Int) {
  mpz_add(Value, Value, Int.Value);
  return *this;
}

void Integer::dump() const { gmp_printf("%Zd", Value); }