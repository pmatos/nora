#pragma once

namespace nir {

// This class represents the void constant.
// https://docs.racket-lang.org/guide/void_undefined.html
class Void {
public:
  Void() = default;
  Void(const Void &V) = default;
  Void(Void &&V) = default;
  Void &operator=(const Void &V) = delete;
  Void &operator=(Void &&V) = default;
  bool operator==(const Void &V) const { return true; }
  ~Void() = default;
};
}; // namespace nir