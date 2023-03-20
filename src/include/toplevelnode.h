#pragma once

#include <variant>

namespace nir {

class Identifier;
class Integer;
class ArithPlus;
class DefineValues;
class Values;
class Void;
class Lambda;
class Begin;

using TLNode = std::variant<Identifier, Integer, ArithPlus, DefineValues,
                            Values, Void, Lambda, Begin>;

}; // namespace nir