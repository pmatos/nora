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
class Begin;
class List;
class Application;
class SetBang;

using ASTNode =
    std::variant<Linklet, Identifier, Integer, ArithPlus, DefineValues, Values,
                 Void, Lambda, Begin, List, Application, SetBang>;

}; // namespace nir