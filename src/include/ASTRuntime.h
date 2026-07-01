#pragma once

#include "AST.h"
#include "Environment.h"

#include <memory>
#include <utility>
#include <vector>

namespace ast {
//
// This file includes the structures that are used in addition to
// those in ast.h during runtime interpretation.
//
// The simplest example is the Closure.

// A Closure is a runtime manifestation of a Lambda.
class Closure : public ClonableNode<Closure, ValueNode> {
public:
  Closure(const Lambda &Lbd, const EnvPtr &Env);
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

// A single continuation mark is a key/value pair. A MarkFrame collects the
// marks belonging to one continuation frame; within a frame each key appears
// at most once (setMark overwrites an existing entry for the same key).
using MarkEntry =
    std::pair<std::unique_ptr<ValueNode>, std::unique_ptr<ValueNode>>;
using MarkFrame = std::vector<MarkEntry>;

// Structural eq?/eqv? approximation used to compare continuation-mark keys.
// Supports the value kinds the interpreter can produce as keys (symbols,
// integers, booleans, characters, strings); everything else compares unequal.
bool valueEq(const ValueNode &A, const ValueNode &B);

// Deep-copy a mark frame.
MarkFrame cloneMarkFrame(const MarkFrame &F);

// Install Key -> Val in Frame, overwriting any existing entry whose key is
// valueEq to Key.
void setMark(MarkFrame &Frame, std::unique_ptr<ValueNode> Key,
             std::unique_ptr<ValueNode> Val);

// The reified result of (current-continuation-marks): a snapshot of the marks
// on the current continuation, one MarkFrame per continuation frame ordered
// innermost (most recent) first.
class ContinuationMarkSet
    : public ClonableNode<ContinuationMarkSet, ValueNode> {
public:
  ContinuationMarkSet() : ClonableNode(ASTNodeKind::AST_ContinuationMarkSet) {}
  explicit ContinuationMarkSet(std::vector<MarkFrame> Frames)
      : ClonableNode(ASTNodeKind::AST_ContinuationMarkSet),
        Frames(std::move(Frames)) {}
  ContinuationMarkSet(const ContinuationMarkSet &Other);

  const std::vector<MarkFrame> &getFrames() const { return Frames; }

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_ContinuationMarkSet;
  }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

private:
  std::vector<MarkFrame> Frames; // innermost (topmost) frame first
};

}; // namespace ast