#include "AnalysisFreeVars.h"

void AnalysisFreeVars::visit(ast::Identifier const &Id) {
  // If the identifier is not in the environment, then it is a free variable.

  for (auto const &Var : llvm::reverse(Vars)) {
    if (Var.count(Id) == 0) {
      Result.insert(Id);
    }
  }
}

void AnalysisFreeVars::visit(ast::Integer const &Int) {
  // Integers do not have free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::Linklet const &Linklet) {
  llvm::errs() << "Free variable analysis only applies to expressions.\n";
}

void AnalysisFreeVars::visit(ast::DefineValues const &DV) {
  llvm::errs() << "Free variable analysis only applies to expressions.\n";
}

void AnalysisFreeVars::visit(ast::Values const &V) {
  // Need to check for free variable in each expression of the Values
  // expression.
  for (auto const &Expr : V.getExprs()) {
    Expr->accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::Void const &Vd) {
  // Void expressions have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::Lambda const &L) {
  const ast::Formal &F = L.getFormals();
  std::set<ast::Identifier> FormalVars;

  if (F.getType() == ast::Formal::Type::Identifier) {
    auto IF = static_cast<const ast::IdentifierFormal &>(F);
    FormalVars.insert(IF.getIdentifier());
  } else if (F.getType() == ast::Formal::Type::List) {
    auto LF = static_cast<const ast::ListFormal &>(F);
    for (auto const &Id : LF.getIds()) {
      FormalVars.insert(Id);
    }
  } else if (F.getType() == ast::Formal::Type::ListRest) {
    auto LRF = static_cast<const ast::ListRestFormal &>(F);
    for (auto const &Id : LRF.getIds()) {
      FormalVars.insert(Id);
    }
    FormalVars.insert(LRF.getRestFormal());
  }

  // Save the current environment.
  Vars.push_back(FormalVars);

  // Check for free variables in the body of the lambda.
  L.getBody().accept(*this);

  // Restore the environment.
  Vars.pop_back();
}

void AnalysisFreeVars::visit(ast::Closure const &L) {
  // Closures by definition do not have free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::Begin const &B) {
  // Iterate through all the begin expressions and check for free variables.
  for (auto const &Expr : B.getBody()) {
    Expr->accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::List const &L) {
  // Iterate through all the List expressions and check for free variables.
  for (auto const &Expr : L.values()) {
    Expr->accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::Application const &A) {
  // Iterate through all the Application expressions and check for free
  // variables.
  for (auto const &Expr : A.getExprs()) {
    Expr->accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::SetBang const &SB) {
  // Check for free variables on the right hand side expression of SetBang
  // expression.
  SB.getExpr().accept(*this);
}

void AnalysisFreeVars::visit(ast::IfCond const &If) {
  // Check for free variables on the condition expression of IfCond expression.
  If.getCond().accept(*this);
  // Check for free variables on the consequent expression of IfCond expression.
  If.getThen().accept(*this);
  // Check for free variables on the alternative expression of IfCond
  // expression.
  If.getElse().accept(*this);
}

void AnalysisFreeVars::visit(ast::BooleanLiteral const &Bool) {
  // Boolean literals have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::LetValues const &LV) {
  std::set<ast::Identifier> LVVars;
  for (size_t Idx = 0; Idx < LV.bindingCount(); Idx++)
    for (auto const &Var : LV.getBindingIds(Idx))
      LVVars.insert(Var);

  Vars.push_back(LVVars);

  for (size_t Idx = 0; Idx < LV.bodyCount(); Idx++)
    LV.getBodyExpr(Idx).accept(*this);

  Vars.pop_back();
}

void AnalysisFreeVars::visit(ast::RuntimeFunction const &LV) {
  // Runtime Functions have no free variables.
  // Nothing to do.
}
