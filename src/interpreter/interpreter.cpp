#include "interpreter.h"

#include <gmp.h>
#include <iostream>
#include <memory>
#include <plog/Log.h>
#include <ranges>
#include <variant>

#include "ast/booleanliteral.h"
#include "ast/formal.h"
#include "ast/linklet.h"
#include "dumper.h"
#include "toplevelnode_inc.h"
#include "utils/downcast.h"
#include "utils/overloaded.h"
#include "utils/upcast.h"
#include "valuenode.h"

#include <llvm/Support/ErrorHandling.h>

// Constructor for interpreter sets up initial environment.
Interpreter::Interpreter() {
  // Add initial environment.
  Envs.emplace_back();
}

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

  // FIXME: why is it that :
  // for (auto &Env : Envs | std::ranges::reverse) {
  // does not work here?
  for (auto Env = Envs.rbegin(); Env != Envs.rend(); ++Env) {
    auto V = Env->lookup(Id);
    if (V) {
      return V;
    }
  }

  // FIXME: error here UndefineIdentifier.
  std::wcerr << "Undefined Identifier: " << Id.getName() << std::endl;
  return nullptr;
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

  // If values contains a single value, it evaluates to that value.
  if (ValuesVec.size() == 1) {
    std::unique_ptr<nir::ValueNode> V =
        downcastExprToValueNode(std::move(ValuesVec[0]));
    return V;
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
  // If there's only one identifier, the variable is assigned the value.
  if (DV.countIds() == 1) {
    Envs.back().add(DV.getIds()[0], std::move(D));
    return std::make_unique<nir::ValueNode>(nir::Void());
  }

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
                              [](nir::Lambda const &L) {
                                return std::make_unique<nir::ValueNode>(L);
                              },
                              [](nir::BooleanLiteral const &Bool) {
                                return std::make_unique<nir::ValueNode>(Bool);
                              },
                              [](auto const &Err) {
                                std::cerr << "Unexpected value in DefineValues."
                                          << std::endl;
                                return std::unique_ptr<nir::ValueNode>();
                              }},
                   E);

    Envs.back().add(Id, std::move(Val));
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
  PLOGD << "Interpreting Lambda: " << std::endl;
  return std::make_unique<nir::ValueNode>(L);
}

std::unique_ptr<nir::ValueNode> Interpreter::operator()(nir::Begin const &B) {
  PLOGD << "Interpreting Begin: " << std::endl;

  // 1. Evaluate each expression in the body.
  std::unique_ptr<nir::ValueNode> D;
  bool First = true;
  for (const auto &BodyExpr : B.getBody()) {
    if (B.isZero()) { // begin0 so only store result of first expression
      if (First) {
        D = std::visit(*this, *BodyExpr);
      } else {
        std::visit(*this, *BodyExpr);
      }
    } else { // normal begin
      D = std::visit(*this, *BodyExpr);
    }

    First = false;
  }

  // 2. Return the stored value.
  return D;
}

std::unique_ptr<nir::ValueNode> Interpreter::operator()(nir::List const &L) {
  PLOGD << "Interpreting List" << std::endl;
  return std::make_unique<nir::ValueNode>(L);
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::Application const &A) {
  // 1. Evaluate the first expression.
  // which should evaluate to a lambda expression.
  std::unique_ptr<nir::ValueNode> D = std::visit(*this, A[0]);
  // Error out if not a lambda expression.
  if (!std::holds_alternative<nir::Lambda>(*D)) {
    std::cerr << "Expected lambda expression in Application." << std::endl;
    return nullptr;
  }
  auto L = std::make_unique<nir::Lambda>(std::get<nir::Lambda>(*D));

  // 2. Evaluate each of the following expressions in order.
  std::vector<std::unique_ptr<nir::ValueNode>> Args;
  Args.reserve(A.length() - 1);
  for (size_t Idx = 1; Idx < A.length(); ++Idx) {
    Args.emplace_back(std::visit(*this, A[Idx]));
  }
  // If we have a list formals then, error out of args diff than formals.
  // If we have a list rest formals then, error out if args less than formals.
  // If it's identifier formals then it does not matter.
  const nir::Formal &F = L->getFormals();
  if (F.getType() == nir::Formal::Type::List) {
    auto LF = static_cast<const nir::ListFormal &>(F);
    if (Args.size() != LF.size()) {
      std::cerr << "Expected " << LF.size() << " arguments, got " << Args.size()
                << std::endl;
      return nullptr;
    }
  } else if (F.getType() == nir::Formal::Type::ListRest) {
    auto LRF = static_cast<const nir::ListRestFormal &>(F);
    if (Args.size() < LRF.size()) {
      std::cerr << "Expected at least " << LRF.size() << " arguments, got "
                << Args.size() << std::endl;
      return nullptr;
    }
  }

  // 3. Apply the lambda expression to the evaluated expressions.
  // Create an environment where each argument is bound to the corresponding
  // value. Then evaluate the lambda body in this environment.
  Environment Env;
  if (F.getType() == nir::Formal::Type::List) {
    auto LF = static_cast<const nir::ListFormal &>(F);
    for (size_t Idx = 0; Idx < Args.size(); ++Idx) {
      Env.add(LF[Idx], std::move(Args[Idx]));
    }
  } else if (F.getType() == nir::Formal::Type::ListRest) {
    auto LRF = static_cast<const nir::ListRestFormal &>(F);
    size_t Idx = 0;
    for (; Idx < LRF.size(); ++Idx) {
      Env.add(LRF[Idx], std::move(Args[Idx]));
    }
    // Create a list of the remaining arguments.
    auto L = std::make_unique<nir::List>();
    for (; Idx < Args.size(); ++Idx) {
      L->appendExpr(std::move(Args[Idx]));
    }
    Env.add(LRF.getRestFormal(), std::make_unique<nir::ValueNode>(*L));
  } else if (F.getType() == nir::Formal::Type::Identifier) {
    auto IF = static_cast<const nir::IdentifierFormal &>(F);
    auto L = std::make_unique<nir::List>();
    for (size_t Idx = 0; Idx < Args.size(); ++Idx) {
      L->appendExpr(std::move(Args[Idx]));
    }
    Env.add(IF.getIdentifier(), std::make_unique<nir::ValueNode>(*L));
  } else {
    llvm_unreachable("unknown formal type");
  }

  Envs.push_back(Env);
  std::unique_ptr<nir::ValueNode> Result = std::visit(*this, L->getBody());
  Envs.pop_back();

  // 4. Return the result of the application.
  return Result;
}

// To interpret a set! expression we set the value of the identifier in the
// current environment and return void.
std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::SetBang const &SB) {
  PLOGD << "Interpreting SetBang\n";

  // 1. Evaluate the expression.
  std::unique_ptr<nir::ValueNode> D = std::visit(*this, SB.getExpr());

  // 2. Set the value of the identifier in the current environment.
  assert(Envs.size() > 0);
  Environment &Env = Envs.back();
  const nir::Identifier &Id = SB.getIdentifier();
  if (!Env.lookup(Id)) {
    std::cerr << "Cannot set undefined identifier." << std::endl;
    return nullptr;
  }
  Env.add(Id, std::move(D));

  // 3. Return void.
  return std::make_unique<nir::ValueNode>(nir::Void());
}

std::unique_ptr<nir::ValueNode> Interpreter::operator()(nir::IfCond const &I) {
  PLOGD << "Interpreting If\n";

  // 1. Evaluate the predicate.
  std::unique_ptr<nir::ValueNode> D = std::visit(*this, I.getCond());

  // 2. If the predicate is false, evaluate the alternative.
  if (std::holds_alternative<nir::BooleanLiteral>(*D) &&
      !std::get<nir::BooleanLiteral>(*D).value()) {
    return std::visit(*this, I.getElse());
  }

  // 3. Everything else evaluates to true, so evaluate the consequent.
  return std::visit(*this, I.getThen());
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::BooleanLiteral const &Bool) {
  PLOGD << "Interpreting BooleanLiteral\n";
  return std::make_unique<nir::ValueNode>(Bool);
}

std::unique_ptr<nir::ValueNode>
Interpreter::operator()(nir::LetValues const &L) {
  PLOGD << "Interpreting LetValues"
        << "\n";
  IF_PLOG(plog::debug) {
    Dumper Dump;
    Dump(L);
    PLOGD << "\n";
  }

  // 1. Evaluate each of the expressions in order.
  std::vector<std::unique_ptr<nir::ValueNode>> ExprValues;
  ExprValues.reserve(L.exprsCount());
  for (size_t Idx = 0; Idx < L.exprsCount(); ++Idx) {
    ExprValues.emplace_back(std::visit(*this, L.getBindingExpr(Idx)));
  }

  // 2. Create an environment where each identifier is bound to the
  // corresponding value. Then evaluate the body in this environment.
  // If the binding variable list has a single identifier, it's simply assigned
  // to the value of the expression. If on the other hand, it's a list of
  // identifiers, then it should have the same length as the values list and
  // each identifier is assigned to the corresponding value.
  PLOGD << "Creating environment for LetValues\n";
  Environment Env;
  for (size_t Idx = 0; Idx < ExprValues.size(); ++Idx) {
    if (std::ranges::size(L.getBindingIds(Idx)) == 1) {
      Env.add(*L.getBindingIds(Idx).begin(), std::move(ExprValues[Idx]));
    } else {
      std::unique_ptr<nir::ValueNode> V = std::move(ExprValues[Idx]);

      if (nir::Values *VsPtr = std::get_if<nir::Values>(V.get())) {
        std::unique_ptr<nir::Values> Vs(VsPtr);
        V.release();

        if (std::ranges::size(L.getBindingIds(Idx)) != Vs->countExprs()) {
          std::cerr << "Expected " << std::ranges::size(L.getBindingIds(Idx))
                    << " values, got " << ExprValues.size() << std::endl;
          return nullptr;
        }

        const auto &IdsExprRange = L.getBindingIds(Idx);
        const auto &ValuesExprRange = Vs->getExprs();

        for (size_t Idx = 0; Idx < Vs->countExprs(); ++Idx) {
          // All the elements in ValuesExprRange are Value, not Expr but we
          // need to downcast them to add them to the environment.
          const auto &E = ValuesExprRange[Idx];
          std::unique_ptr<nir::ValueNode> Val = std::visit(
              overloaded{
                  [](nir::Void const &V) {
                    return std::make_unique<nir::ValueNode>(V);
                  },
                  [](nir::Integer const &I) {
                    return std::make_unique<nir::ValueNode>(I);
                  },
                  [](nir::Values const &V) {
                    return std::make_unique<nir::ValueNode>(V);
                  },
                  [](nir::Lambda const &L) {
                    return std::make_unique<nir::ValueNode>(L);
                  },
                  [](nir::BooleanLiteral const &Bool) {
                    return std::make_unique<nir::ValueNode>(Bool);
                  },
                  [](auto const &Err) {
                    std::cerr
                        << "Unexpected value in interpretation of let-values."
                        << std::endl;
                    return std::unique_ptr<nir::ValueNode>();
                  }},
              E);

          PLOGD << "Adding " << std::wstring(IdsExprRange[Idx].getName())
                << " to environment\n";
          Env.add(IdsExprRange[Idx], std::move(Val));
        }

      } else {
        Dumper Dump;
        std::cerr << "Expected a values node, got ";
        std::visit(Dump, *V);
        return nullptr;
      }
    }
  }
  PLOGD << " Pushing new environment for LetValues\n";
  Envs.push_back(Env);

  PLOGD << "Evaluating body of LetValues\n";
  std::unique_ptr<nir::ValueNode> Result = nullptr;
  for (size_t I = 0; I < L.exprsCount(); ++I) {
    Result = std::visit(*this, L.getBodyExpr(I));
  }

  Envs.pop_back();

  // 3. Return the result of the let-values expression.
  return Result;
}