#pragma once

#include "ast.h"
#include "environment.h"

#include <memory>

namespace ast {
//
// This file includes the structures that are used in addition to
// those in ast.h during runtime interpretation.
//
// The simplest example is the Closure.

// A Closure is a runtime manifestation of a Lambda.
class Closure : public ClonableNode<Closure, ValueNode> {
public:
  Closure(const Lambda &Lbd, const std::vector<Environment> &Envs);
  Closure(const Closure &Other);

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Closure;
  }

  void dump() const override;
  void write() const override;

  const Lambda &getLambda() const { return *L; }
  const Environment &getEnvironment() const { return Env; }

private:
  std::unique_ptr<Lambda> L;
  Environment Env;
};

}; // namespace ast