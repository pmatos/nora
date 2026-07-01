#pragma once

#include "AST.h"
#include "Environment.h"

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

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  const Lambda &getLambda() const { return *L; }
  const Environment &getEnvironment() const { return Env; }

private:
  std::unique_ptr<Lambda> L;
  Environment Env;
};

// A CaseLambdaClosure is the runtime manifestation of a CaseLambda.
class CaseLambdaClosure : public ClonableNode<CaseLambdaClosure, ValueNode> {
public:
  CaseLambdaClosure(const CaseLambda &CL, const std::vector<Environment> &Envs);
  CaseLambdaClosure(const CaseLambdaClosure &Other);

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_CaseLambdaClosure;
  }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  const CaseLambda &getCaseLambda() const { return *CL; }
  const Environment &getEnvironment() const { return Env; }

private:
  std::unique_ptr<CaseLambda> CL;
  Environment Env;
};

}; // namespace ast