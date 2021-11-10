#pragma once

#include <variant>

namespace nir {

class Identifier;
class Integer;
class ArithPlus;
class DefineValues;
class Values;
class Linklet;
class Void;
class Lambda;

using ASTNode = std::variant<Linklet, Identifier, Integer, ArithPlus,
                             DefineValues, Values, Void, Lambda>;

}; // namespace nir