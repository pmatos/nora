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

void AnalysisFreeVars::visit(ast::CaseLambda const &CL) {
  // A case-lambda's free variables are the union over all its clauses. Each
  // clause is a Lambda that binds its own formals around its own body.
  for (size_t Idx = 0; Idx < CL.size(); ++Idx) {
    CL[Idx].accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::Closure const &L) {
  // Closures by definition do not have free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::CaseLambdaClosure const &CL) {
  // Case-lambda closures by definition do not have free variables.
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
  if (L.getTail()) {
    L.getTail()->accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::Vector const &Vec) {
  // Iterate through all the Vector elements and check for free variables.
  for (auto const &Expr : Vec.values()) {
    Expr->accept(*this);
  }
}

void AnalysisFreeVars::visit(ast::VariableReference const &VR) {
  // A variable reference is an opaque value with no free variables.
  // Nothing to do.
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

void AnalysisFreeVars::visit(ast::Char const &C) {
  // Characters have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::String const &Str) {
  // Strings have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::LetValues const &LV) {
  std::set<ast::Identifier> LVVars;
  for (size_t Idx = 0; Idx < LV.bindingCount(); Idx++)
    for (auto const &Var : LV.getBindingIds(Idx))
      LVVars.insert(Var);

  // In let-values the binding expressions are evaluated in the enclosing scope,
  // so the bound identifiers do not shadow them. In letrec-values the
  // identifiers are in scope for the binding expressions too, enabling
  // recursion, so those references must not be reported as free.
  if (!LV.isRec())
    for (size_t Idx = 0; Idx < LV.bindingCount(); Idx++)
      LV.getBindingExpr(Idx).accept(*this);

  Vars.push_back(LVVars);

  if (LV.isRec())
    for (size_t Idx = 0; Idx < LV.bindingCount(); Idx++)
      LV.getBindingExpr(Idx).accept(*this);

  for (size_t Idx = 0; Idx < LV.bodyCount(); Idx++)
    LV.getBodyExpr(Idx).accept(*this);

  Vars.pop_back();
}

void AnalysisFreeVars::visit(ast::RuntimeFunction const &LV) {
  // Runtime Functions have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::QuotedExpr const &QE) {
  // Quoted Expressions have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::Symbol const &Sym) {
  // Symbols have no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::WithContinuationMark const &WCM) {
  // Check for free variables in the key, value and result expressions.
  WCM.getKey().accept(*this);
  WCM.getVal().accept(*this);
  WCM.getResult().accept(*this);
}

void AnalysisFreeVars::visit(ast::ContinuationMarkSet const &CMS) {
  // Continuation mark sets are runtime values with no free variables.
  // Nothing to do.
}

void AnalysisFreeVars::visit(ast::Keyword const &K) {
  // Keywords have no free variables.
  // Nothing to do.
}