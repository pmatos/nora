#include "ast/identifier.h"

using namespace nir;

Identifier::Identifier(std::wstring_view Id) : Id(Id) {}

Identifier &Identifier::operator=(Identifier &&I) noexcept {
  Id = I.Id;
  return *this;
}

const std::wstring_view Identifier::getName() const { return Id; }
