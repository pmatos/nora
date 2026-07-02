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

// A Closure is a runtime manifestation of a Lambda. It captures the lexical
// environment (the shared scope chain) in which the lambda was evaluated, so
// that set! and letrec forward references resolve through the live bindings.
class Closure : public ClonableNode<Closure, ValueNode> {
public:
  Closure(const Lambda &Lbd, EnvPtr Env);
  Closure(const Closure &Other);

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Closure;
  }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  const Lambda &getLambda() const { return *L; }
  const EnvPtr &getEnv() const { return Env; }

private:
  std::unique_ptr<Lambda> L;
  EnvPtr Env;
};

// A CaseLambdaClosure is the runtime manifestation of a CaseLambda. Like
// Closure, it captures the lexical environment shared across all its clauses.
class CaseLambdaClosure : public ClonableNode<CaseLambdaClosure, ValueNode> {
public:
  CaseLambdaClosure(const CaseLambda &CL, EnvPtr Env);
  CaseLambdaClosure(const CaseLambdaClosure &Other);

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_CaseLambdaClosure;
  }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  const CaseLambda &getCaseLambda() const { return *CL; }
  const EnvPtr &getEnv() const { return Env; }

private:
  std::unique_ptr<CaseLambda> CL;
  EnvPtr Env;
};

// A box is a mutable single-slot cell. Its cell is heap-allocated and shared:
// copying a Box (which the interpreter does on every environment lookup) shares
// the same cell, so a set-box! through one reference is visible through all of
// them and (eq? b b) holds. This is the first piece of M2's shared, identity-
// bearing value model; the cell moves onto the GC heap in a later M2 slice.
class Box : public ClonableNode<Box, ValueNode> {
public:
  explicit Box(std::unique_ptr<ValueNode> V);
  Box(const Box &Other); // shares the cell (shallow copy)
  ~Box() = default;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Box;
  }

  // Current contents as a fresh value (following the interpreter's value
  // model); used by unbox.
  std::unique_ptr<ValueNode> get() const;
  // Replace the shared cell's contents (set-box!). Const because the Box
  // wrapper is immutable; the cell it references is not.
  void set(std::unique_ptr<ValueNode> V) const;
  // Cell identity for eq?: two Box values are eq? iff they share a cell.
  const void *identity() const { return C.get(); }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

private:
  struct Cell {
    std::unique_ptr<ValueNode> Value;
  };
  std::shared_ptr<Cell> C;
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