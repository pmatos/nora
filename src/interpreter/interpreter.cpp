#include "interpreter.h"

#include <gmp.h>
#include <iostream>
#include <memory>
#include <plog/Log.h>
#include <variant>

#include "ast/linklet.h"
#include "toplevelnode_inc.h"
#include "utils/overloaded.h"
#include "utils/upcast.h"
#include "valuenode.h"

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::Identifier const &Id) {
  // Check if there's a binding for Id in environment,
  // if so, return the value.
  // If not, error UndefinedIdentifier.
  PLOGD << "Interpreting Identifier: ";
  IF_PLOG(plog::debug) {
    std::wstring Name(Id.getName());
    PLOGD << Name << std::endl;
  }

  if (Env.find(Id) != Env.end()) {
    auto V = *(Env[Id]);
    return std::make_unique<nir::ValueNode>(V);
  } else {
    return nullptr;
  }
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::Integer const &Int) {
  PLOGD << "Interpreting Integer: ";
  IF_PLOG(plog::debug) {
    Int.dump();
    PLOGD << std::endl;
  }

  return std::make_unique<nir::ValueNode>(Int);
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::Linklet const &Linklet) {
  PLOGD << "Interpreting Linklet" << std::endl;
  std::unique_ptr<nir::ValueNode> D;
  for (const auto &BodyForm : Linklet.getBody()) {
    D = std::visit(*this, *BodyForm);
  }
  return D; // returns the value of the last form in the linklet body.
}

std::unique_ptr<nir::ValueNode> Interpreter::operator()(nir::Values const &V) {
  PLOGD << "Interpreting Values: " << std::endl;
  std::vector<std::unique_ptr<nir::ExprNode>> ValuesVec;
  for (const auto &Expr : V.getExprs()) {
    std::unique_ptr<nir::ValueNode> D = std::visit(*this, *Expr);
    ValuesVec.emplace_back(upcastNode(D));
  }
  std::unique_ptr<nir::ValueNode> Values =
      std::make_unique<nir::ValueNode>(nir::Values(std::move(ValuesVec)));
  return Values;
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::DefineValues const &DV) {
  PLOGD << "Interpreting DefineValues: " << std::endl;

  // 1. Evaluate values body.
  std::unique_ptr<nir::ValueNode> D = std::visit(*this, DV.getBody());
  if (!D) {
    return nullptr;
  }

  // 2. Check number of values and number of identifiers match.

  // Check if the expression is a Values.
  if (!std::holds_alternative<nir::Values>(*D)) {
    std::cerr << "Expected Values in DefineValues." << std::endl;
    return nullptr;
  }
  const nir::Values &V = std::get<nir::Values>(*D);

  // Check if the number of values is equal to the number of identifiers.
  if (DV.countIds() != V.countExprs()) {
    std::cerr << "Expected " << DV.countIds() << " values, got "
              << V.countExprs() << std::endl;
    return nullptr;
  }

  // 2. Add bindings to the environment.
  size_t Idx = 0;
  for (const auto &Id : DV.getIds()) {
    const nir::ExprNode &E = V.getExprs()[Idx++];
    // Because we know that ValuesExpr is a value we can down cast in place.
    std::unique_ptr<nir::ValueNode> Val =
        std::visit(overloaded{[](nir::Void const &V) {
                                return std::make_unique<nir::ValueNode>(V);
                              },
                              [](nir::Integer const &I) {
                                return std::make_unique<nir::ValueNode>(I);
                              },
                              [](nir::Values const &V) {
                                return std::make_unique<nir::ValueNode>(V);
                              },
                              [](auto const &Err) {
                                std::cerr << "Unexpected value in DefineValues."
                                          << std::endl;
                                return std::unique_ptr<nir::ValueNode>();
                              }},
                   E);

    Env[Id] = std::move(Val);
  }

  // Return void.
  return std::make_unique<nir::ValueNode>(nir::Void());
}

std::unique_ptr<nir::ValueNode> Interpreter::operator()(nir::Void const &Vd) {
  PLOGD << "Interpreting Void" << std::endl;
  return std::make_unique<nir::ValueNode>(nir::Void());
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::ArithPlus const &AP) {
  PLOGD << "Interpreting ArithPlus: " << std::endl;

  nir::Integer Sum(0);

  for (auto const &Arg : AP.getArgs()) {
    std::unique_ptr<nir::ValueNode> D = std::visit(*this, *Arg);
    if (std::holds_alternative<nir::Integer>(*D)) {
      Sum += std::get<nir::Integer>(*D);
    } else {
      std::cerr << "Unexpected value in ArithPlus." << std::endl;
      return nullptr;
    }
  }
  return std::make_unique<nir::ValueNode>(Sum);
}

std::unique_ptr<nir::ValueNode> Interpreter::operator()(nir::Lambda const &L) {
  return nullptr; // TODO
}
