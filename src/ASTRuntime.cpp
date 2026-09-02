#include "ASTRuntime.h"

#include "Casting.h"

#include <iostream>
#include <utility>

using namespace ast;

// A closure captures the lexical environment (shared scope chain) in which the
// lambda was evaluated. Capturing the live chain - rather than a copy of the
// free variables - gives correct lexical scoping (a free variable unbound at
// the definition site stays unbound) and lets letrec forward/mutual references
// resolve once the shared scope is filled in.
Closure::Closure(const Lambda &Lbd, EnvPtr Env)
    : ClonableNode(ASTNodeKind::AST_Closure),
      L(std::unique_ptr<Lambda>(static_cast<Lambda *>(Lbd.clone()))),
      Env(std::move(Env)) {}

Closure::Closure(const Closure &Other)
    : ClonableNode(ASTNodeKind::AST_Closure),
      L(std::unique_ptr<Lambda>(static_cast<Lambda *>(Other.L->clone()))),
      Env(Other.Env) {}

void Closure::dump() const {
  // TODO: Implement.
  llvm::dbgs() << "<closure: not implemented>\n";
}
void Closure::write() const {}

CaseLambdaClosure::CaseLambdaClosure(const CaseLambda &CLbd, EnvPtr Env)
    : ClonableNode(ASTNodeKind::AST_CaseLambdaClosure),
      CL(std::unique_ptr<CaseLambda>(static_cast<CaseLambda *>(CLbd.clone()))),
      Env(std::move(Env)) {}

CaseLambdaClosure::CaseLambdaClosure(const CaseLambdaClosure &Other)
    : ClonableNode(ASTNodeKind::AST_CaseLambdaClosure),
      CL(std::unique_ptr<CaseLambda>(
          static_cast<CaseLambda *>(Other.CL->clone()))),
      Env(Other.Env) {}

void CaseLambdaClosure::dump() const {
  llvm::dbgs() << "<case-lambda closure: not implemented>\n";
}
void CaseLambdaClosure::write() const {}

Box::Box(std::unique_ptr<ValueNode> V)
    : ClonableNode(ASTNodeKind::AST_Box), C(std::make_shared<Cell>()) {
  C->Value = std::move(V);
}

Box::Box(const Box &Other) : ClonableNode(ASTNodeKind::AST_Box), C(Other.C) {}

std::unique_ptr<ValueNode> Box::get() const {
  return std::unique_ptr<ValueNode>(C->Value->clone());
}

void Box::set(std::unique_ptr<ValueNode> V) const { C->Value = std::move(V); }

void Box::dump() const { llvm::dbgs() << "#&<box>\n"; }

void Box::write() const {
  std::cout << "#&";
  C->Value->write();
}

Pair::Pair(std::unique_ptr<ValueNode> Car, std::unique_ptr<ValueNode> Cdr)
    : ClonableNode(ASTNodeKind::AST_Pair), C(std::make_shared<Cell>()) {
  C->Car = std::move(Car);
  C->Cdr = std::move(Cdr);
}

Pair::Pair(const Pair &Other)
    : ClonableNode(ASTNodeKind::AST_Pair), C(Other.C) {}

std::unique_ptr<ValueNode> Pair::car() const {
  return std::unique_ptr<ValueNode>(C->Car->clone());
}

std::unique_ptr<ValueNode> Pair::cdr() const {
  return std::unique_ptr<ValueNode>(C->Cdr->clone());
}

void Pair::setCar(std::unique_ptr<ValueNode> V) const { C->Car = std::move(V); }
void Pair::setCdr(std::unique_ptr<ValueNode> V) const { C->Cdr = std::move(V); }

void Pair::dump() const { llvm::dbgs() << "#<pair>\n"; }

void Pair::write() const {
  std::cout << "(";
  C->Car->write();
  std::cout << " . ";
  C->Cdr->write();
  std::cout << ")";
}

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
