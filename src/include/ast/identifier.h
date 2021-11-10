#pragma once

#include <compare>
#include <string_view>

class IdPool;

namespace nir {

class Identifier {
public:
  Identifier() = delete;
  Identifier(const Identifier &) = default;
  Identifier(Identifier &&) = default;
  Identifier &operator=(const Identifier &) = delete;
  Identifier &operator=(Identifier &&I) noexcept;
  ~Identifier() = default;

  auto operator<=>(const Identifier &I) const = default;

  [[nodiscard]] const std::wstring_view getName() const;

private:
  friend class ::IdPool;
  explicit Identifier(std::wstring_view Id);
  std::wstring_view Id;
};

} // namespace nir