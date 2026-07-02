#include "Interpreter.h"

#include <llvm/ADT/Twine.h>
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
//
// Runtime errors are reported through the DiagnosticEngine (anchored at the
// offending node's source location) and unwind the current form via abortEval;
// main() checks Diag.hadError() to decide the exit status.

Interpreter::Interpreter(nora::DiagnosticEngine &Diag) : Diag(Diag) {}

Interpreter::~Interpreter() {
  // Break closure/scope reference cycles so the shared_ptr scopes are actually
  // reclaimed (LeakSanitizer would otherwise flag them). A closure captures the
  // scope chain it was defined in, and that scope may in turn bind the closure
  // (a top-level lambda in GlobalEnv, or a letrec clause), so dropping every
  // scope's bindings severs the cycle before the scopes are destroyed.
  for (auto &S : AllScopes) {
    if (S) {
      S->Vars.clear();
    }
  }
}

EnvPtr Interpreter::newScope(const EnvPtr &Parent) {
  auto S = std::make_shared<Scope>();
  S->Parent = Parent;
  AllScopes.push_back(S);
  return S;
}

// Bind one let-values / letrec-values clause into Vars: a single identifier
// takes the whole value, while several identifiers require a Values result
// whose arity matches. Returns false (after reporting) on a mismatch.
static bool bindValues(nora::DiagnosticEngine &Diag, llvm::SMLoc Loc,
                       Environment &Vars, const ast::LetValues::IdRange &Ids,
                       std::unique_ptr<ast::ValueNode> Val) {
  if (std::ranges::size(Ids) == 1) {
    Vars.add(Ids[0], std::move(Val));
    return true;
  }

  auto Vs = dyn_castU<ast::Values>(Val);
  if (!Vs) {
    Diag.error(Loc, "let-values binding expected multiple values");
    return false;
  }
  const size_t Expected = std::ranges::size(Ids);
  if (Expected != Vs->countExprs()) {
    Diag.error(Loc, llvm::Twine("let-values binding expected ") +
                        llvm::Twine(Expected) + " values, got " +
                        llvm::Twine(Vs->countExprs()));
    return false;
  }

  const auto &ValuesExprRange = Vs->getExprs();
  for (size_t Idx = 0; Idx < Vs->countExprs(); ++Idx) {
    // Elements of a Values node are values stored as exprs; downcast in place.
    std::unique_ptr<ast::ExprNode> EPtr(ValuesExprRange[Idx].clone());
    Vars.add(Ids[Idx], dyn_castU<ast::ValueNode>(EPtr));
  }
  return true;
}

// Returns true if the formals F accept NArgs supplied arguments.
static bool formalsAccept(const ast::Formal &F, size_t NArgs) {
  switch (F.getType()) {
  case ast::Formal::Type::List:
    return static_cast<const ast::ListFormal &>(F).size() == NArgs;
  case ast::Formal::Type::ListRest:
    return static_cast<const ast::ListRestFormal &>(F).size() <= NArgs;
  case ast::Formal::Type::Identifier:
    return true;
  }
  llvm_unreachable("unknown formal type");
}

//
// Driver
//

void Interpreter::visit(ast::Linklet const &Linklet) {
  // The top-level scope persists across body forms so that later
  // define-values are visible to earlier closures at call time.
  GlobalEnv = newScope(nullptr);

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
    if (Diag.hadError()) {
      break;
    }
  }
  Result = std::move(Last);
}

void Interpreter::run() {
  while (true) {
    if (Kont.size() > PeakKont) {
      PeakKont = Kont.size();
    }
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
      const bool IsLast = Top.Idx + 1 == Top.Exprs.size();
      if (IsLast && !Top.Begin0) {
        // Tail position: drop the sequence frame before its final expression
        // (mirrors IfBranch) so a tail call there reuses the enclosing
        // activation frame rather than stacking a new one.
        const ast::ExprNode *E = Top.Exprs[Top.Idx];
        EnvPtr SeqEnv = Top.Env;
        Kont.pop_back();
        Control = E;
        Env = SeqEnv;
        M = Mode::Eval;
      } else {
        Control = Top.Exprs[Top.Idx];
        Env = Top.Env;
        Top.Idx++;
        M = Mode::Eval;
      }
    } else {
      // Only begin0 reaches here: its frame persists to the end to return the
      // saved first value; a plain sequence's final expression is handled
      // above.
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
      llvm::SMLoc AppLoc = Top.AppLoc;
      llvm::SMLoc OpLoc = Top.Exprs[0]->getLoc();
      Kont.pop_back();
      applyProcedure(std::move(Vals), AppLoc, OpLoc);
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

    // let-values: all binding expressions were evaluated in the enclosing
    // environment; only now are the identifiers bound, in a fresh scope.
    EnvPtr ScopePtr = newScope(OuterEnv);
    for (size_t I = 0; I < Vals.size(); ++I) {
      if (!bindValues(Diag, Let->getLoc(), ScopePtr->Vars,
                      Let->getBindingIds(I), std::move(Vals[I]))) {
        abortEval();
        return;
      }
    }

    llvm::SmallVector<const ast::ExprNode *> Body;
    for (size_t I = 0; I < Let->bodyCount(); ++I) {
      Body.push_back(&Let->getBodyExpr(I));
    }
    evalBody(std::move(Body), ScopePtr);
    break;
  }

  case Frame::LetRec: {
    // letrec-values: the recursive scope is already in scope; bind the value
    // just produced, then evaluate the next binding expression (or the body)
    // in that same scope so forward/mutual references resolve.
    const ast::LetValues *Let = Top.Let;
    EnvPtr RecScope = Top.RecScope;
    if (!bindValues(Diag, Let->getLoc(), RecScope->Vars,
                    Let->getBindingIds(Top.Idx), std::move(Val))) {
      abortEval();
      return;
    }
    Top.Idx++;
    if (Top.Idx < Let->exprsCount()) {
      Control = &Let->getBindingExpr(Top.Idx);
      Env = RecScope;
      M = Mode::Eval;
      break;
    }

    Kont.pop_back();
    llvm::SmallVector<const ast::ExprNode *> Body;
    for (size_t I = 0; I < Let->bodyCount(); ++I) {
      Body.push_back(&Let->getBodyExpr(I));
    }
    evalBody(std::move(Body), RecScope);
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
    if (!Vs) {
      Diag.error(DV->getLoc(),
                 "define-values expected multiple values from its body");
      abortEval();
      break;
    }
    if (Vs->countExprs() != DV->countIds()) {
      Diag.error(DV->getLoc(), llvm::Twine("define-values expected ") +
                                   llvm::Twine(DV->countIds()) +
                                   " values, got " +
                                   llvm::Twine(Vs->countExprs()));
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
      Diag.error(Id->getLoc(), llvm::Twine("cannot set unbound identifier: ") +
                                   Id->getName());
      abortEval();
      break;
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
    // Push a dedicated mark-bearing frame holding this key/value, and evaluate
    // the result expression under it. The frame is popped when the result
    // produces a value (see the WcmMark case), so the mark's dynamic extent is
    // exactly the result expression - a with-continuation-mark in non-tail
    // position no longer leaks its mark into later expressions.
    Kont.emplace_back(Frame::WcmMark);
    ast::setMark(Kont.back().Marks, std::move(KeyV), std::move(ValV));
    Control = ResultE;
    Env = E;
    M = Mode::Eval;
    break;
  }

  case Frame::WcmMark: {
    // The result expression has produced a value; discard the mark frame and
    // pass the value through to the enclosing continuation.
    std::unique_ptr<ast::ValueNode> V = std::move(Val);
    Kont.pop_back();
    deliver(std::move(V));
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
    std::vector<std::unique_ptr<ast::ValueNode>> Vals, llvm::SMLoc AppLoc,
    llvm::SMLoc OpLoc) {
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
      Diag.error(AppLoc, llvm::Twine("invalid arguments to '") + Name + "'");
      abortEval();
      return;
    }
    deliver(std::move(R));
    return;
  }

  // Select the clause to apply: a plain closure has one, a case-lambda picks
  // the first clause whose formals accept the argument count.
  const ast::Lambda *Clause = nullptr;
  EnvPtr Captured;

  if (auto *C = llvm::dyn_cast<ast::Closure>(Op.get())) {
    const ast::Formal &F = C->getLambda().getFormals();
    if (F.getType() == ast::Formal::Type::List) {
      size_t N = static_cast<const ast::ListFormal &>(F).size();
      if (NArgs != N) {
        Diag.error(AppLoc, llvm::Twine("arity mismatch: expected ") +
                               llvm::Twine(N) + " argument(s), got " +
                               llvm::Twine(NArgs));
        abortEval();
        return;
      }
    } else if (F.getType() == ast::Formal::Type::ListRest) {
      size_t N = static_cast<const ast::ListRestFormal &>(F).size();
      if (NArgs < N) {
        Diag.error(AppLoc, llvm::Twine("arity mismatch: expected at least ") +
                               llvm::Twine(N) + " argument(s), got " +
                               llvm::Twine(NArgs));
        abortEval();
        return;
      }
    }
    Clause = &C->getLambda();
    Captured = C->getEnv();
  } else if (auto *CLC = llvm::dyn_cast<ast::CaseLambdaClosure>(Op.get())) {
    const ast::CaseLambda &CL = CLC->getCaseLambda();
    for (size_t Idx = 0; Idx < CL.size(); ++Idx) {
      if (formalsAccept(CL[Idx].getFormals(), NArgs)) {
        Clause = &CL[Idx];
        break;
      }
    }
    if (!Clause) {
      Diag.error(AppLoc, llvm::Twine("case-lambda: no matching clause for ") +
                             llvm::Twine(NArgs) + " argument(s)");
      abortEval();
      return;
    }
    Captured = CLC->getEnv();
  } else {
    Diag.error(OpLoc, "application: expected a procedure in operator position");
    abortEval();
    return;
  }

  // Build the callee environment: a fresh argument scope whose parent is the
  // closure's captured lexical environment (proper lexical scoping).
  const ast::Formal &F = Clause->getFormals();
  EnvPtr CalleeScope = newScope(Captured);
  switch (F.getType()) {
  case ast::Formal::Type::List: {
    auto LF = static_cast<const ast::ListFormal &>(F);
    for (size_t I = 0; I < NArgs; ++I) {
      CalleeScope->Vars.add(LF[I], std::move(Vals[I + 1]));
    }
    break;
  }
  case ast::Formal::Type::ListRest: {
    auto LRF = static_cast<const ast::ListRestFormal &>(F);
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

  // Tail call: if the enclosing continuation frame is the caller's own
  // activation (Frame::Call), reuse it instead of stacking a new one. Together
  // with popping Seq/if/let-body frames before their tail sub-expression, this
  // makes self- and mutual tail recursion run in O(1) continuation space.
  if (!Kont.empty() && Kont.back().K == Frame::Call) {
    Frame &Enc = Kont.back();
    Enc.Callee = std::move(Op); // frees the previous activation's closure
    Enc.Marks.clear();          // the reused frame begins a fresh activation
    Control = &Clause->getBody();
    Env = CalleeScope;
    M = Mode::Eval;
    return;
  }

  Kont.emplace_back(Frame::Call);
  // The Call frame takes ownership of the closure so its (cloned) lambda body,
  // into which Control now points, outlives this function.
  Kont.back().Callee = std::move(Op);
  Control = &Clause->getBody();
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

  Diag.error(Id.getLoc(), llvm::Twine("unbound identifier: ") + Id.getName());
  abortEval();
}

void Interpreter::visit(ast::Application const &A) {
  Kont.emplace_back(Frame::App);
  Frame &F = Kont.back();
  F.Env = Env;
  F.AppLoc = A.getLoc();
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

  if (L.isRec()) {
    // letrec-values: the bound identifiers are in scope while their own
    // binding expressions are evaluated. Create the recursive scope up front
    // and fill it in left to right.
    EnvPtr RecScope = newScope(Env);
    if (NB == 0) {
      llvm::SmallVector<const ast::ExprNode *> Body;
      for (size_t I = 0; I < L.bodyCount(); ++I) {
        Body.push_back(&L.getBodyExpr(I));
      }
      evalBody(std::move(Body), RecScope);
      return;
    }
    Kont.emplace_back(Frame::LetRec);
    Frame &F = Kont.back();
    F.Let = &L;
    F.RecScope = RecScope;
    F.Idx = 0;
    Control = &L.getBindingExpr(0);
    Env = RecScope;
    M = Mode::Eval;
    return;
  }

  if (NB == 0) {
    EnvPtr ScopePtr = newScope(Env);
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

void Interpreter::visit(ast::CaseLambda const &CL) {
  deliver(std::make_unique<ast::CaseLambdaClosure>(CL, Env));
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

void Interpreter::visit(ast::Keyword const &K) {
  deliver(std::unique_ptr<ast::ValueNode>(K.clone()));
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

void Interpreter::visit(ast::VariableReference const &VR) {
  deliver(std::unique_ptr<ast::ValueNode>(VR.clone()));
}

void Interpreter::visit(ast::Closure const &C) {
  deliver(std::unique_ptr<ast::ValueNode>(C.clone()));
}

void Interpreter::visit(ast::CaseLambdaClosure const &C) {
  deliver(std::unique_ptr<ast::ValueNode>(C.clone()));
}

void Interpreter::visit(ast::RuntimeFunction const &RF) {
  deliver(std::unique_ptr<ast::ValueNode>(RF.clone()));
}

void Interpreter::visit(ast::ContinuationMarkSet const &CMS) {
  deliver(std::unique_ptr<ast::ValueNode>(CMS.clone()));
}
