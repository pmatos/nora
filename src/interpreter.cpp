#include "interpreter.h"

#include "Casting.h"

#include <llvm/Support/ErrorHandling.h>

#include <gmp.h>
#include <iostream>
#include <memory>
#include <ranges>

#include "Casting.h"
#include "ast_fwd.h"
#include "llvm/Support/raw_ostream.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Interpreter"

// Constructor for interpreter sets up initial environment.
Interpreter::Interpreter() {
  // Add initial environment.
  Envs.emplace_back();
}

bool Interpreter::isBound(const ast::Identifier &Id) const {
  // FIXME: the same questions as in Interpreter::visit(ast::Identifier const
  // &Id) regarding the use of std::ranges::reverse applies.
  for (auto Env = Envs.rbegin(); Env != Envs.rend(); ++Env) {
    auto V = Env->lookup(Id);
    if (V) {
      return true;
    }
  }
  return false;
}

void Interpreter::visit(ast::Identifier const &Id) {
  // Check if there's a binding for Id in environment,
  // if so, return the value.
  // If not, error UndefinedIdentifier.
  LLVM_DEBUG({
    llvm::dbgs() << "Interpreting Identifier: ";
    std::string Name(Id.getName());
    llvm::dbgs() << Name << "\n";
  });

  // FIXME: why is it that :
  // for (auto &Env : Envs | std::ranges::reverse) {
  // does not work here?
  for (auto Env = Envs.rbegin(); Env != Envs.rend(); ++Env) {
    auto V = Env->lookup(Id);
    if (V) {
      Result = std::move(V);
      return;
    }
  }

  // It's not in the environment, but is it an identifier that's
  // part of the runtime?
  const std::string Name(Id.getName());
  if (Runtime::getInstance().isRuntimeFunction(Name)) {
    LLVM_DEBUG(llvm::dbgs() << "Identifier is runtime function\n");
    std::unique_ptr<ast::RuntimeFunction> RF =
        Runtime::getInstance().lookupRuntimeFunction(Name);
    Result = std::move(RF);
    return;
  }

  // FIXME: error here UndefineIdentifier.
  llvm::errs() << "Undefined Identifier: " << Id.getName() << "\n";
  Result = nullptr;
}

void Interpreter::visit(ast::RuntimeFunction const &LV) {
  Result = std::unique_ptr<ast::ValueNode>(LV.clone());
}

void Interpreter::visit(ast::Integer const &Int) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting Integer: " << Int.asString()
                          << "\n");
  Result = std::unique_ptr<ast::ValueNode>(Int.clone());
  assert(llvm::dyn_cast<ast::Integer>(Result.get()));
}

void Interpreter::visit(ast::Linklet const &Linklet) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting Linklet\n");

  for (const auto &BodyForm : Linklet.getBody()) {
    // The result of the last expression ends up being saved in Result.
    BodyForm->accept(*this);
  }
}

void Interpreter::visit(ast::Values const &V) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting Values\n");
  std::vector<std::unique_ptr<ast::ValueNode>> ValuesVec;
  for (const auto &Expr : V.getExprs()) {
    Expr->accept(*this);
    assert(Result);
    ValuesVec.emplace_back(std::move(Result));
  }

  // If values contains a single value, it evaluates to that value.
  if (ValuesVec.size() == 1) {
    Result = std::move(ValuesVec[0]);
    return;
  }

  std::vector<std::unique_ptr<ast::ExprNode>> Exprs;
  Exprs.reserve(ValuesVec.size());
  for (auto &V : ValuesVec) {
    Exprs.emplace_back(std::move(V));
  }

  std::unique_ptr<ast::ValueNode> Vs(new ast::Values(std::move(Exprs)));
  Result = std::move(Vs);
}

void Interpreter::visit(ast::DefineValues const &DV) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting DefineValues\n");

  // 1. Evaluate values body.
  DV.getBody().accept(*this);
  if (!Result) {
    return;
  }

  // 2. Check number of values and number of identifiers match.
  // If there's only one identifier, the variable is assigned the value.
  if (DV.countIds() == 1) {
    Envs.back().add(DV.getIds()[0], std::move(Result));
    Result = std::unique_ptr<ast::ValueNode>(new ast::Void());
    return;
  }

  // Check if the expression is a Values.
  if (!llvm::isa<ast::Values>(*Result)) {
    llvm::errs() << "Expected Values in DefineValues.\n";
    Result = nullptr;
    return;
  }
  auto const &V = dyn_castU<ast::Values>(Result);

  // Check if the number of values is equal to the number of identifiers.
  if (DV.countIds() != V->countExprs()) {
    llvm::errs() << "Expected " << DV.countIds() << " values, got "
                 << V->countExprs() << "\n";
    Result = nullptr;
    return;
  }

  // 2. Add bindings to the environment.
  size_t Idx = 0;
  for (const auto &Id : DV.getIds()) {
    const auto &Exprs = V->getExprs();
    const ast::ExprNode &E = Exprs[Idx++];

    // Because we know that ValuesExpr is a value we can down cast in place.
    std::unique_ptr<ast::ExprNode> EPtr(E.clone());
    std::unique_ptr<ast::ValueNode> Val = dyn_castU<ast::ValueNode>(EPtr);
    assert(Val && "Expected Value in ValuesExpr.");
    Envs.back().add(Id, std::move(Val));
  }

  // Return void.
  Result = std::unique_ptr<ast::ValueNode>(new ast::Void());
}

void Interpreter::visit(ast::Void const &Vd) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting Void\n");
  Result = std::unique_ptr<ast::ValueNode>(new ast::Void());
}

void Interpreter::visit(ast::Lambda const &L) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting Lambda\n");
  Result = std::unique_ptr<ast::ValueNode>(L.clone());
}

void Interpreter::visit(ast::Begin const &B) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting Begin\n");

  // 1. Evaluate each expression in the body.
  std::unique_ptr<ast::ValueNode> D;
  bool First = true;
  for (const auto &BodyExpr : B.getBody()) {
    if (B.isZero()) { // begin0 so only store result of first expression
      if (First) {
        BodyExpr->accept(*this);
        D = std::move(Result);
      } else {
        BodyExpr->accept(*this);
      }
    } else { // normal begin
      BodyExpr->accept(*this);
      D = std::move(Result);
    }

    First = false;
  }

  // 2. Return the stored value.
  Result = std::move(D);
}

void Interpreter::visit(ast::List const &L) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting List\n");
  Result = std::unique_ptr<ast::ValueNode>(L.clone());
}

void Interpreter::visit(ast::Application const &A) {
  // 1. Evaluate the first expression.
  // which should evaluate to a lambda expression.
  A[0].accept(*this);
  std::unique_ptr<ast::ValueNode> D = std::move(Result);

  // Error out if not a lambda expression or runtime expression.
  std::unique_ptr<ast::Lambda> L = dyn_castU<ast::Lambda>(D);
  if (!L) {
    // maybe a runtime function?
    std::unique_ptr<ast::RuntimeFunction> RF =
        dyn_castU<ast::RuntimeFunction>(D);
    if (!RF) {
      std::cerr << "Expected lambda expression in Application.\n";
      return;
    }

    // OK - it's a runtime function. Lets prepare the arguments to call it.
    // This is a weird situation where we need two vectors although it doesn't
    // make much sense. We need to move the Results out of Result and into
    // ArgHolder. ArgHolder will keep the ownership of all results. Then we need
    // to create another vector to pass to the runtime function. This vector
    // will contain pointers to the results in ArgHolder. This sucks a bit but
    // at this point, I am not sure if there's a point in focusing on optimizing
    // this.
    std::vector<std::unique_ptr<ast::ValueNode>> ArgHolder;
    std::vector<const ast::ValueNode *> Args;
    ArgHolder.reserve(A.length() - 1);
    Args.reserve(A.length() - 1);
    for (size_t Idx = 1; Idx < A.length(); ++Idx) {
      A[Idx].accept(*this);
      assert(Result && "Expected result from expression.");
      ArgHolder.emplace_back(std::move(Result));
      Args.emplace_back(ArgHolder.back().get());
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Calling runtime function: " << RF->getName() << "\n";
      for (const ast::ValueNode *Arg : Args) {
        assert(llvm::dyn_cast<ast::Integer>(Arg) &&
               "Expected Integer in runtime function call.");
        llvm::dbgs() << "  Arg: ";
        Arg->dump();
        llvm::dbgs() << "\n";
      }
    });

    // Call the runtime function.
    Result = Runtime::getInstance().callFunction(RF->getName(), Args);
    return;
  }

  // 2. Evaluate each of the following expressions in order.
  std::vector<std::unique_ptr<ast::ValueNode>> Args;
  Args.reserve(A.length() - 1);
  for (size_t Idx = 1; Idx < A.length(); ++Idx) {
    A[Idx].accept(*this);
    assert(Result && "Expected result from expression.");
    Args.emplace_back(std::move(Result));
  }
  // If we have a list formals then, error out of args diff than formals.
  // If we have a list rest formals then, error out if args less than formals.
  // If it's identifier formals then it does not matter.
  const ast::Formal &F = L->getFormals();
  if (F.getType() == ast::Formal::Type::List) {
    auto LF = static_cast<const ast::ListFormal &>(F);
    if (Args.size() != LF.size()) {
      std::cerr << "Expected " << LF.size() << " arguments, got " << Args.size()
                << std::endl;
      Result = nullptr;
      return;
    }
  } else if (F.getType() == ast::Formal::Type::ListRest) {
    auto LRF = static_cast<const ast::ListRestFormal &>(F);
    if (Args.size() < LRF.size()) {
      std::cerr << "Expected at least " << LRF.size() << " arguments, got "
                << Args.size() << std::endl;
      Result = nullptr;
      return;
    }
  }

  // 3. Apply the lambda expression to the evaluated expressions.
  // Create an environment where each argument is bound to the corresponding
  // value. Then evaluate the lambda body in this environment.
  Environment Env;
  if (F.getType() == ast::Formal::Type::List) {
    auto LF = static_cast<const ast::ListFormal &>(F);
    for (size_t Idx = 0; Idx < Args.size(); ++Idx) {
      Env.add(LF[Idx], std::move(Args[Idx]));
    }
  } else if (F.getType() == ast::Formal::Type::ListRest) {
    auto LRF = static_cast<const ast::ListRestFormal &>(F);
    size_t Idx = 0;
    for (; Idx < LRF.size(); ++Idx) {
      Env.add(LRF[Idx], std::move(Args[Idx]));
    }
    // Create a list of the remaining arguments.
    auto L = std::make_unique<ast::List>();
    for (; Idx < Args.size(); ++Idx) {
      L->appendExpr(std::move(Args[Idx]));
    }

    Env.add(LRF.getRestFormal(), std::unique_ptr<ast::ValueNode>(L->clone()));
  } else if (F.getType() == ast::Formal::Type::Identifier) {
    auto IF = static_cast<const ast::IdentifierFormal &>(F);
    auto L = std::make_unique<ast::List>();
    for (size_t Idx = 0; Idx < Args.size(); ++Idx) {
      L->appendExpr(std::move(Args[Idx]));
    }
    Env.add(IF.getIdentifier(), std::unique_ptr<ast::ValueNode>(L->clone()));
  } else {
    llvm_unreachable("unknown formal type");
  }

  Envs.push_back(Env);
  // 4. Return the result of the application.
  L->getBody().accept(*this);
  Envs.pop_back();
}

// To interpret a set! expression we set the value of the identifier in the
// current environment and return void.
void Interpreter::visit(ast::SetBang const &SB) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting SetBang\n");

  // 1. Evaluate the expression.
  SB.getExpr().accept(*this);
  std::unique_ptr<ast::ValueNode> D = std::move(Result);

  // 2. Set the value of the identifier in the current environment.
  assert(Envs.size() > 0);
  Environment &Env = Envs.back();
  const ast::Identifier &Id = SB.getIdentifier();
  if (!Env.lookup(Id)) {
    std::cerr << "Cannot set undefined identifier." << std::endl;
    return;
  }
  Env.add(Id, std::move(D));

  // 3. Return void.
  Result = std::unique_ptr<ast::ValueNode>(new ast::Void());
}

void Interpreter::visit(ast::IfCond const &I) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting If\n");

  // 1. Evaluate the predicate.
  I.getCond().accept(*this);
  std::unique_ptr<ast::ValueNode> D = std::move(Result);

  // 2. If the predicate is false, evaluate the alternative.
  std::unique_ptr<ast::BooleanLiteral> Bool = dyn_castU<ast::BooleanLiteral>(D);
  if (Bool && !Bool->value()) {
    I.getElse().accept(*this);
    return;
  }

  // 3. Everything else evaluates to true, so evaluate the consequent.
  I.getThen().accept(*this);
}

void Interpreter::visit(ast::BooleanLiteral const &Bool) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting BooleanLiteral\n");
  Result = std::unique_ptr<ast::ValueNode>(Bool.clone());
}

void Interpreter::visit(ast::LetValues const &L) {
  LLVM_DEBUG(llvm::dbgs() << "Interpreting LetValues\n");
  LLVM_DEBUG(L.dump(); llvm::dbgs() << "\n";);

  // 1. Evaluate each of the expressions in order.
  std::vector<std::unique_ptr<ast::ValueNode>> ExprValues;
  ExprValues.reserve(L.exprsCount());
  for (size_t Idx = 0; Idx < L.exprsCount(); ++Idx) {
    L.getBindingExpr(Idx).accept(*this);
    assert(Result);
    ExprValues.emplace_back(std::move(Result));
  }

  // 2. Create an environment where each identifier is bound to the
  // corresponding value. Then evaluate the body in this environment.
  // If the binding variable list has a single identifier, it's simply assigned
  // to the value of the expression. If on the other hand, it's a list of
  // identifiers, then it should have the same length as the values list and
  // each identifier is assigned to the corresponding value.
  LLVM_DEBUG(llvm::dbgs() << "Creating environment for LetValues\n");
  Environment Env;
  for (size_t Idx = 0; Idx < ExprValues.size(); ++Idx) {
    if (std::ranges::size(L.getBindingIds(Idx)) == 1) {
      Env.add(*L.getBindingIds(Idx).begin(), std::move(ExprValues[Idx]));
    } else {
      std::unique_ptr<ast::ValueNode> V = std::move(ExprValues[Idx]);

      if (auto const &Vs = dyn_castU<ast::Values>(V)) {
        if (std::ranges::size(L.getBindingIds(Idx)) != Vs->countExprs()) {
          std::cerr << "Expected " << std::ranges::size(L.getBindingIds(Idx))
                    << " values, got " << ExprValues.size() << std::endl;
          return;
        }

        const auto &IdsExprRange = L.getBindingIds(Idx);
        const auto &ValuesExprRange = Vs->getExprs();

        for (size_t Idx = 0; Idx < Vs->countExprs(); ++Idx) {
          // All the elements in ValuesExprRange are Value,
          // not Expr but we need to downcast them to add
          // them to the environment.
          const auto &E = ValuesExprRange[Idx];
          std::unique_ptr<ast::ExprNode> EPtr(E.clone());
          std::unique_ptr<ast::ValueNode> Val = dyn_castU<ast::ValueNode>(EPtr);

          LLVM_DEBUG(llvm::dbgs()
                     << "Adding " << std::string(IdsExprRange[Idx].getName())
                     << " to environment\n");
          Env.add(IdsExprRange[Idx], std::move(Val));
        }

      } else {
        std::cerr << "Expected a values node, got ";
        V->dump();
        return;
      }
    }
  }
  LLVM_DEBUG(llvm::dbgs() << " Pushing new environment for LetValues\n");
  Envs.push_back(Env);

  LLVM_DEBUG(llvm::dbgs() << "Evaluating body of LetValues\n");
  for (size_t I = 0; I < L.exprsCount(); ++I) {
    // 3. Return the result of the let-values expression.
    L.getBodyExpr(I).accept(*this);
  }

  Envs.pop_back();
}