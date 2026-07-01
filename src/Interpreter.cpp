#include "Interpreter.h"

#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include "ASTRuntime.h"
#include "Casting.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "Interpreter"

// This file implements a CEK/CESK abstract machine. The visit() methods are the
// Eval-mode transitions: given the expression in Control, each either produces
// a value (deliver + switch to Continue mode) or decomposes into a
// subexpression plus a continuation frame (push a Frame + stay in Eval mode).
// continueStep() is the Continue-mode transition: it delivers the value
// register to the top continuation frame. visit(Linklet) is the driver.

Interpreter::Interpreter() = default;

//
// Driver
//

void Interpreter::visit(ast::Linklet const &Linklet) {
  // The top-level scope persists across body forms so that later
  // define-values are visible to earlier closures at call time.
  GlobalEnv = std::make_shared<Scope>();

  std::unique_ptr<ast::ValueNode> Last;
  for (const auto &BodyForm : Linklet.getBody()) {
    Kont.clear();
    Kont.emplace_back(Frame::Halt);
    Control = BodyForm.get();
    Env = GlobalEnv;
    Val = nullptr;
    M = Mode::Eval;
    run();
    Last = std::move(Val);
  }
  Result = std::move(Last);
}

void Interpreter::run() {
  while (true) {
    if (M == Mode::Eval) {
      Control->accept(*this);
    } else {
      if (Kont.back().K == Frame::Halt) {
        return; // Val holds the result (or null on error).
      }
      continueStep();
    }
  }
}

//
// Continue-mode transition
//

void Interpreter::continueStep() {
  Frame &Top = Kont.back();
  switch (Top.K) {
  case Frame::Halt:
    // Handled by run(); should not reach here.
    return;

  case Frame::Seq: {
    if (Top.Begin0 && Top.Idx == 1) {
      Top.Saved = std::move(Val);
    }
    if (Top.Idx < Top.Exprs.size()) {
      Control = Top.Exprs[Top.Idx];
      Env = Top.Env;
      Top.Idx++;
      M = Mode::Eval;
    } else {
      std::unique_ptr<ast::ValueNode> R =
          Top.Begin0 ? std::move(Top.Saved) : std::move(Val);
      Kont.pop_back();
      deliver(std::move(R));
    }
    break;
  }

  case Frame::IfBranch: {
    const ast::ExprNode *ThenE = Top.ThenE;
    const ast::ExprNode *ElseE = Top.ElseE;
    EnvPtr E = Top.Env;
    std::unique_ptr<ast::ValueNode> Cond = std::move(Val);
    Kont.pop_back();
    auto *B = llvm::dyn_cast_or_null<ast::BooleanLiteral>(Cond.get());
    Control = (B && !B->value()) ? ElseE : ThenE;
    Env = E;
    M = Mode::Eval;
    break;
  }

  case Frame::App: {
    Top.Done.push_back(std::move(Val));
    if (Top.Done.size() < Top.Exprs.size()) {
      Control = Top.Exprs[Top.Done.size()];
      Env = Top.Env;
      M = Mode::Eval;
    } else {
      std::vector<std::unique_ptr<ast::ValueNode>> Vals = std::move(Top.Done);
      Kont.pop_back();
      applyProcedure(std::move(Vals));
    }
    break;
  }

  case Frame::MkValues: {
    Top.Done.push_back(std::move(Val));
    if (Top.Done.size() < Top.Exprs.size()) {
      Control = Top.Exprs[Top.Done.size()];
      Env = Top.Env;
      M = Mode::Eval;
    } else {
      std::vector<std::unique_ptr<ast::ValueNode>> Vals = std::move(Top.Done);
      Kont.pop_back();
      if (Vals.size() == 1) {
        deliver(std::move(Vals[0]));
      } else {
        llvm::SmallVector<std::unique_ptr<ast::ExprNode>> Exprs;
        Exprs.reserve(Vals.size());
        for (auto &Vv : Vals) {
          Exprs.emplace_back(std::move(Vv));
        }
        deliver(std::make_unique<ast::Values>(std::move(Exprs)));
      }
    }
    break;
  }

  case Frame::LetBind: {
    Top.Done.push_back(std::move(Val));
    const ast::LetValues *Let = Top.Let;
    if (Top.Done.size() < Let->exprsCount()) {
      Control = &Let->getBindingExpr(Top.Done.size());
      Env = Top.Env;
      M = Mode::Eval;
      break;
    }

    std::vector<std::unique_ptr<ast::ValueNode>> Vals = std::move(Top.Done);
    EnvPtr OuterEnv = Top.Env;
    Kont.pop_back();

    auto ScopePtr = std::make_shared<Scope>();
    ScopePtr->Parent = OuterEnv;
    bool Ok = true;
    for (size_t I = 0; I < Vals.size(); ++I) {
      auto Ids = Let->getBindingIds(I);
      size_t NIds = std::ranges::size(Ids);
      if (NIds == 1) {
        ScopePtr->Vars.add(*Ids.begin(), std::move(Vals[I]));
        continue;
      }
      auto *Vs = llvm::dyn_cast<ast::Values>(Vals[I].get());
      if (!Vs || Vs->countExprs() != NIds) {
        std::cerr << "Expected " << NIds << " values in let-values binding"
                  << std::endl;
        Ok = false;
        break;
      }
      size_t J = 0;
      for (auto const &Id : Ids) {
        const ast::ExprNode &E = Vs->getExprs()[J++];
        std::unique_ptr<ast::ExprNode> EP(E.clone());
        std::unique_ptr<ast::ValueNode> Vv = dyn_castU<ast::ValueNode>(EP);
        ScopePtr->Vars.add(Id, std::move(Vv));
      }
    }
    if (!Ok) {
      abortEval();
      break;
    }

    llvm::SmallVector<const ast::ExprNode *> Body;
    for (size_t I = 0; I < Let->bodyCount(); ++I) {
      Body.push_back(&Let->getBodyExpr(I));
    }
    evalBody(std::move(Body), ScopePtr);
    break;
  }

  case Frame::Define: {
    const ast::DefineValues *DV = Top.Def;
    EnvPtr DefEnv = Top.DefEnv;
    std::unique_ptr<ast::ValueNode> V = std::move(Val);
    Kont.pop_back();

    if (DV->countIds() == 1) {
      DefEnv->Vars.add(DV->getIds()[0], std::move(V));
      deliver(std::make_unique<ast::Void>());
      break;
    }
    auto *Vs = llvm::dyn_cast_or_null<ast::Values>(V.get());
    if (!Vs || Vs->countExprs() != DV->countIds()) {
      llvm::errs() << "Expected " << DV->countIds() << " values in "
                   << "define-values.\n";
      abortEval();
      break;
    }
    size_t J = 0;
    for (auto const &Id : DV->getIds()) {
      const ast::ExprNode &E = Vs->getExprs()[J++];
      std::unique_ptr<ast::ExprNode> EP(E.clone());
      std::unique_ptr<ast::ValueNode> Vv = dyn_castU<ast::ValueNode>(EP);
      DefEnv->Vars.add(Id, std::move(Vv));
    }
    deliver(std::make_unique<ast::Void>());
    break;
  }

  case Frame::Set: {
    const ast::Identifier *Id = Top.SetId;
    EnvPtr E = Top.Env;
    std::unique_ptr<ast::ValueNode> V = std::move(Val);
    Kont.pop_back();
    if (!envSet(E, *Id, std::move(V))) {
      llvm::errs() << "Cannot set undefined identifier.\n";
    }
    deliver(std::make_unique<ast::Void>());
    break;
  }

  case Frame::WcmKey: {
    const ast::ExprNode *ValE = Top.WcmValE;
    const ast::ExprNode *ResultE = Top.WcmResultE;
    EnvPtr E = Top.Env;
    std::unique_ptr<ast::ValueNode> KeyV = std::move(Val);
    Kont.pop_back();
    Kont.emplace_back(Frame::WcmVal);
    Frame &WV = Kont.back();
    WV.WcmResultE = ResultE;
    WV.WcmKeyV = std::move(KeyV);
    WV.Env = E;
    Control = ValE;
    Env = E;
    M = Mode::Eval;
    break;
  }

  case Frame::WcmVal: {
    const ast::ExprNode *ResultE = Top.WcmResultE;
    EnvPtr E = Top.Env;
    std::unique_ptr<ast::ValueNode> KeyV = std::move(Top.WcmKeyV);
    std::unique_ptr<ast::ValueNode> ValV = std::move(Val);
    Kont.pop_back();
    // Install the mark on the frame that encloses the with-continuation-mark
    // form. There is always such a frame (a Halt frame sits at the bottom of
    // every top-level form's continuation).
    ast::setMark(Kont.back().Marks, std::move(KeyV), std::move(ValV));
    Control = ResultE;
    Env = E;
    M = Mode::Eval;
    break;
  }

  case Frame::Call: {
    // The activation's body has produced a value; its frame (and marks) is
    // discarded and the value flows to the caller's continuation.
    std::unique_ptr<ast::ValueNode> V = std::move(Val);
    Kont.pop_back();
    deliver(std::move(V));
    break;
  }
  }
}

//
// Application
//

void Interpreter::applyProcedure(
    std::vector<std::unique_ptr<ast::ValueNode>> Vals) {
  std::unique_ptr<ast::ValueNode> Op = std::move(Vals[0]);
  const size_t NArgs = Vals.size() - 1;

  if (!Op) {
    abortEval();
    return;
  }

  if (auto *RF = llvm::dyn_cast<ast::RuntimeFunction>(Op.get())) {
    const std::string &Name = RF->getName();
    // (current-continuation-marks) needs the machine's continuation, so it is
    // handled here rather than as a plain runtime function.
    if (Name == "current-continuation-marks") {
      deliver(std::make_unique<ast::ContinuationMarkSet>(snapshotMarks()));
      return;
    }
    llvm::SmallVector<const ast::ValueNode *> Args;
    Args.reserve(NArgs);
    for (size_t I = 1; I < Vals.size(); ++I) {
      Args.push_back(Vals[I].get());
    }
    std::unique_ptr<ast::ValueNode> R =
        Runtime::getInstance().callFunction(Name, Args);
    if (!R) {
      abortEval();
      return;
    }
    deliver(std::move(R));
    return;
  }

  auto *C = llvm::dyn_cast<ast::Closure>(Op.get());
  if (!C) {
    llvm::errs()
        << "Expected closure or runtime function expression in application.\n";
    abortEval();
    return;
  }

  const ast::Lambda &L = C->getLambda();
  const ast::Formal &F = L.getFormals();

  // Build the callee environment lexically: the persistent top-level scope
  // (so later top-level define-values remain visible), extended with the
  // closure's captured bindings, extended with the argument bindings. The
  // caller's environment is deliberately NOT part of the chain - splicing it in
  // would leak caller locals into the callee and let a free variable that is
  // unbound at the closure's definition site resolve to a same-named binding at
  // the call site (dynamic scoping) instead of raising an unbound-identifier
  // error.
  EnvPtr AfterClosure = envExtend(GlobalEnv, C->getEnvironment());
  auto CalleeScope = std::make_shared<Scope>();
  CalleeScope->Parent = AfterClosure;

  switch (F.getType()) {
  case ast::Formal::Type::List: {
    auto LF = static_cast<const ast::ListFormal &>(F);
    if (NArgs != LF.size()) {
      std::cerr << "Expected " << LF.size() << " arguments, got " << NArgs
                << std::endl;
      abortEval();
      return;
    }
    for (size_t I = 0; I < NArgs; ++I) {
      CalleeScope->Vars.add(LF[I], std::move(Vals[I + 1]));
    }
    break;
  }
  case ast::Formal::Type::ListRest: {
    auto LRF = static_cast<const ast::ListRestFormal &>(F);
    if (NArgs < LRF.size()) {
      std::cerr << "Expected at least " << LRF.size() << " arguments, got "
                << NArgs << std::endl;
      abortEval();
      return;
    }
    size_t I = 0;
    for (; I < LRF.size(); ++I) {
      CalleeScope->Vars.add(LRF[I], std::move(Vals[I + 1]));
    }
    auto Rest = std::make_unique<ast::List>();
    for (; I < NArgs; ++I) {
      Rest->appendExpr(std::move(Vals[I + 1]));
    }
    CalleeScope->Vars.add(LRF.getRestFormal(), std::move(Rest));
    break;
  }
  case ast::Formal::Type::Identifier: {
    auto IF = static_cast<const ast::IdentifierFormal &>(F);
    auto Lst = std::make_unique<ast::List>();
    for (size_t I = 0; I < NArgs; ++I) {
      Lst->appendExpr(std::move(Vals[I + 1]));
    }
    CalleeScope->Vars.add(IF.getIdentifier(), std::move(Lst));
    break;
  }
  }

  Kont.emplace_back(Frame::Call);
  // The Call frame takes ownership of the closure so its lambda body (into
  // which Control now points) outlives this function.
  Kont.back().Callee = std::move(Op);
  Control = &L.getBody();
  Env = CalleeScope;
  M = Mode::Eval;
}

void Interpreter::evalBody(llvm::SmallVector<const ast::ExprNode *> Body,
                           const EnvPtr &E) {
  assert(!Body.empty() && "empty body sequence");
  if (Body.size() == 1) {
    Control = Body[0];
    Env = E;
    M = Mode::Eval;
    return;
  }
  Kont.emplace_back(Frame::Seq);
  Frame &F = Kont.back();
  F.Env = E;
  F.Begin0 = false;
  F.Exprs = std::move(Body);
  F.Idx = 1;
  Control = F.Exprs[0];
  Env = E;
  M = Mode::Eval;
}

std::vector<ast::MarkFrame> Interpreter::snapshotMarks() const {
  std::vector<ast::MarkFrame> Frames;
  Frames.reserve(Kont.size());
  for (auto It = Kont.rbegin(); It != Kont.rend(); ++It) {
    Frames.push_back(ast::cloneMarkFrame(It->Marks));
  }
  return Frames;
}

bool Interpreter::isBound(const ast::Identifier &Id) const {
  if (GlobalEnv && envLookup(GlobalEnv, Id)) {
    return true;
  }
  return Runtime::getInstance().isRuntimeFunction(std::string(Id.getName()));
}

//
// Eval-mode transitions
//

void Interpreter::visit(ast::Identifier const &Id) {
  if (auto V = envLookup(Env, Id)) {
    deliver(std::move(V));
    return;
  }

  const std::string Name(Id.getName());
  if (Runtime::getInstance().isRuntimeFunction(Name)) {
    deliver(Runtime::getInstance().lookupRuntimeFunction(Name));
    return;
  }

  llvm::errs() << "Undefined Identifier: " << Id.getName() << "\n";
  abortEval();
}

void Interpreter::visit(ast::Application const &A) {
  Kont.emplace_back(Frame::App);
  Frame &F = Kont.back();
  F.Env = Env;
  F.Exprs.reserve(A.length());
  for (size_t I = 0; I < A.length(); ++I) {
    F.Exprs.push_back(&A[I]);
  }
  Control = F.Exprs[0];
  M = Mode::Eval;
}

void Interpreter::visit(ast::Begin const &B) {
  const auto &Body = B.getBody();
  if (Body.size() == 1) {
    Control = Body[0].get();
    M = Mode::Eval;
    return;
  }
  Kont.emplace_back(Frame::Seq);
  Frame &F = Kont.back();
  F.Env = Env;
  F.Begin0 = B.isZero();
  F.Exprs.reserve(Body.size());
  for (auto const &E : Body) {
    F.Exprs.push_back(E.get());
  }
  F.Idx = 1;
  Control = F.Exprs[0];
  M = Mode::Eval;
}

void Interpreter::visit(ast::IfCond const &I) {
  Kont.emplace_back(Frame::IfBranch);
  Frame &F = Kont.back();
  F.Env = Env;
  F.ThenE = &I.getThen();
  F.ElseE = &I.getElse();
  Control = &I.getCond();
  M = Mode::Eval;
}

void Interpreter::visit(ast::Values const &V) {
  const size_t N = V.countExprs();
  if (N == 0) {
    llvm::SmallVector<std::unique_ptr<ast::ExprNode>> Empty;
    deliver(std::make_unique<ast::Values>(std::move(Empty)));
    return;
  }
  Kont.emplace_back(Frame::MkValues);
  Frame &F = Kont.back();
  F.Env = Env;
  F.Exprs.reserve(N);
  for (size_t I = 0; I < N; ++I) {
    F.Exprs.push_back(&V.getExprs()[I]);
  }
  Control = F.Exprs[0];
  M = Mode::Eval;
}

void Interpreter::visit(ast::LetValues const &L) {
  const size_t NB = L.exprsCount();
  if (NB == 0) {
    auto ScopePtr = std::make_shared<Scope>();
    ScopePtr->Parent = Env;
    llvm::SmallVector<const ast::ExprNode *> Body;
    for (size_t I = 0; I < L.bodyCount(); ++I) {
      Body.push_back(&L.getBodyExpr(I));
    }
    evalBody(std::move(Body), ScopePtr);
    return;
  }
  Kont.emplace_back(Frame::LetBind);
  Frame &F = Kont.back();
  F.Env = Env;
  F.Let = &L;
  Control = &L.getBindingExpr(0);
  M = Mode::Eval;
}

void Interpreter::visit(ast::DefineValues const &DV) {
  Kont.emplace_back(Frame::Define);
  Frame &F = Kont.back();
  F.Env = Env;
  F.Def = &DV;
  F.DefEnv = Env;
  Control = &DV.getBody();
  M = Mode::Eval;
}

void Interpreter::visit(ast::SetBang const &SB) {
  Kont.emplace_back(Frame::Set);
  Frame &F = Kont.back();
  F.Env = Env;
  F.SetId = &SB.getIdentifier();
  Control = &SB.getExpr();
  M = Mode::Eval;
}

void Interpreter::visit(ast::WithContinuationMark const &WCM) {
  Kont.emplace_back(Frame::WcmKey);
  Frame &F = Kont.back();
  F.Env = Env;
  F.WcmValE = &WCM.getVal();
  F.WcmResultE = &WCM.getResult();
  Control = &WCM.getKey();
  M = Mode::Eval;
}

void Interpreter::visit(ast::Lambda const &L) {
  deliver(std::make_unique<ast::Closure>(L, Env));
}

void Interpreter::visit(ast::Integer const &Int) {
  deliver(std::unique_ptr<ast::ValueNode>(Int.clone()));
}

void Interpreter::visit(ast::BooleanLiteral const &Bool) {
  deliver(std::unique_ptr<ast::ValueNode>(Bool.clone()));
}

void Interpreter::visit(ast::Char const &C) {
  deliver(std::unique_ptr<ast::ValueNode>(C.clone()));
}

void Interpreter::visit(ast::String const &Str) {
  deliver(std::unique_ptr<ast::ValueNode>(Str.clone()));
}

void Interpreter::visit(ast::Symbol const &Sym) {
  deliver(std::unique_ptr<ast::ValueNode>(Sym.clone()));
}

void Interpreter::visit(ast::List const &L) {
  deliver(std::unique_ptr<ast::ValueNode>(L.clone()));
}

void Interpreter::visit(ast::Vector const &Vec) {
  deliver(std::unique_ptr<ast::ValueNode>(Vec.clone()));
}

void Interpreter::visit(ast::Void const &Vd) {
  deliver(std::unique_ptr<ast::ValueNode>(Vd.clone()));
}

void Interpreter::visit(ast::QuotedExpr const &QE) {
  deliver(std::unique_ptr<ast::ValueNode>(QE.clone()));
}

void Interpreter::visit(ast::Closure const &C) {
  deliver(std::unique_ptr<ast::ValueNode>(C.clone()));
}

void Interpreter::visit(ast::RuntimeFunction const &RF) {
  deliver(std::unique_ptr<ast::ValueNode>(RF.clone()));
}

void Interpreter::visit(ast::ContinuationMarkSet const &CMS) {
  deliver(std::unique_ptr<ast::ValueNode>(CMS.clone()));
}
