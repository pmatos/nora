#include "ASTRuntime.h"

#include "AnalysisFreeVars.h"
#include "Casting.h"

#include <iostream>
#include <utility>

using namespace ast;

Closure::Closure(const Lambda &Lbd, const EnvPtr &Env)
    : ClonableNode(ASTNodeKind::AST_Closure),
      L(std::unique_ptr<Lambda>(static_cast<Lambda *>(Lbd.clone()))) {

  // To create a closure we need to:

  // 1. Find the free variables in the lambda.
  AnalysisFreeVars AFV;
  L->accept(AFV);
  auto const &FreeVars = AFV.getResult();

  // 2. Find in the current environment the values of the free variables and
  // save a copy of each into the closure's captured environment.
  for (auto const &Var : FreeVars) {
    if (auto Val = envLookup(Env, Var)) {
      this->Env.add(Var, std::move(Val));
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

void Closure::dump() const {
  // TODO: Implement.
  llvm::dbgs() << "<closure: not implemented>\n";
}
void Closure::write() const {}

//
// Continuation marks
//

bool ast::valueEq(const ValueNode &A, const ValueNode &B) {
  // A quoted datum such as 'k evaluates to a QuotedExpr wrapping the datum;
  // compare the underlying data so 'k is eq? to 'k.
  if (const auto *QA = llvm::dyn_cast<QuotedExpr>(&A)) {
    return valueEq(QA->getQuotedExpr(), B);
  }
  if (const auto *QB = llvm::dyn_cast<QuotedExpr>(&B)) {
    return valueEq(A, QB->getQuotedExpr());
  }
  if (A.getKind() != B.getKind()) {
    return false;
  }
  switch (A.getKind()) {
  case ASTNode::ASTNodeKind::AST_Symbol:
    return llvm::cast<Symbol>(A) == llvm::cast<Symbol>(B);
  case ASTNode::ASTNodeKind::AST_Integer:
    return llvm::cast<Integer>(A) == llvm::cast<Integer>(B);
  case ASTNode::ASTNodeKind::AST_BooleanLiteral:
    return llvm::cast<BooleanLiteral>(A).value() ==
           llvm::cast<BooleanLiteral>(B).value();
  case ASTNode::ASTNodeKind::AST_Char:
    return llvm::cast<Char>(A).getValue() == llvm::cast<Char>(B).getValue();
  case ASTNode::ASTNodeKind::AST_String:
    return llvm::cast<String>(A).getValue() == llvm::cast<String>(B).getValue();
  default:
    return false;
  }
}

MarkFrame ast::cloneMarkFrame(const MarkFrame &F) {
  MarkFrame Out;
  Out.reserve(F.size());
  for (auto const &E : F) {
    Out.emplace_back(std::unique_ptr<ValueNode>(E.first->clone()),
                     std::unique_ptr<ValueNode>(E.second->clone()));
  }
  return Out;
}

void ast::setMark(MarkFrame &Frame, std::unique_ptr<ValueNode> Key,
                  std::unique_ptr<ValueNode> Val) {
  for (auto &E : Frame) {
    if (valueEq(*E.first, *Key)) {
      E.second = std::move(Val);
      return;
    }
  }
  Frame.emplace_back(std::move(Key), std::move(Val));
}

ContinuationMarkSet::ContinuationMarkSet(const ContinuationMarkSet &Other)
    : ClonableNode(ASTNodeKind::AST_ContinuationMarkSet) {
  Frames.reserve(Other.Frames.size());
  for (auto const &F : Other.Frames) {
    Frames.emplace_back(cloneMarkFrame(F));
  }
}

void ContinuationMarkSet::dump() const {
  llvm::dbgs() << "#<continuation-mark-set>";
}
void ContinuationMarkSet::write() const {
  std::cout << "#<continuation-mark-set>";
}
