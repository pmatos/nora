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
class List;
class Application;
class SetBang;
class IfCond;
class BooleanLiteral;

using TLNode = std::variant<Identifier, Integer, ArithPlus, DefineValues,
                            Values, Void, Lambda, Begin, List, Application,
                            SetBang, IfCond, BooleanLiteral>;

}; // namespace nir