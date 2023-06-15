#include "ASTRuntime.h"

#include "AnalysisFreeVars.h"

#include <utility>

using namespace ast;

Closure::Closure(const Lambda &Lbd, const std::vector<Environment> &Envs)
    : ClonableNode(ASTNodeKind::AST_Closure),
      L(std::unique_ptr<Lambda>(static_cast<Lambda *>(Lbd.clone()))) {

  // To create a closure we need to:

  // 1. Find the free variables in the lambda.
  AnalysisFreeVars AFV;
  L->accept(AFV);
  auto const &FreeVars = AFV.getResult();

  // 2. Find in the current environment, the values of the free variables
  // and save them.
  for (auto const &Var : FreeVars) {
    for (auto const &E : llvm::reverse(Envs)) {
      auto const &Val = E.lookup(Var);
      if (Val) {
        Env.add(Var, std::unique_ptr<ValueNode>(Val->clone()));
        break;
      }
    }
  }
}

Closure::Closure(const Closure &Other)
    : ClonableNode(ASTNodeKind::AST_Closure),
      L(std::unique_ptr<Lambda>(static_cast<Lambda *>(Other.L->clone()))) {
  for (auto const &E : Other.Env) {
    Env.add(E.first, std::unique_ptr<ValueNode>(E.second->clone()));
  }
}

void Closure::dump() const {}
void Closure::write() const {}