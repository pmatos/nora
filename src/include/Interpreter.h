#pragma once

// This interpreter is a CEK/CESK-style abstract machine. Instead of walking the
// AST recursively on the host C++ stack, the machine makes the continuation an
// explicit, first-class data structure: a stack of typed frames (Kont). A step
// either evaluates an expression (Eval mode) - decomposing it into a
// subexpression plus a frame that remembers "what to do with the value" - or
// delivers a value to the top frame (Continue mode).
//
// Each continuation frame carries a set of continuation marks. This is how
// Racket models marks, and it makes with-continuation-mark and the mark query
// primitives fall out naturally. Making the continuation explicit is also the
// foundation for the delimited-continuation primitives (prompts, call/cc,
// composable continuations) tracked as follow-ups to issue #11.

#include <cassert>
#include <memory>
#include <vector>

#include <llvm/ADT/SmallVector.h>

#include "AST.h"
#include "ASTRuntime.h"
#include "ASTVisitor.h"
#include "Diagnostics.h"
#include "Environment.h"
#include "Runtime.h"

class Interpreter : public ASTVisitor {
public:
  explicit Interpreter(nora::DiagnosticEngine &Diag);
  ~Interpreter() override;

  // The visit methods below are single-step Eval transitions of the machine,
  // except visit(Linklet), which is the driver that runs the machine to
  // completion for each top-level form.
  // Note: keep the list sorted alphabetically.
  virtual void visit(ast::Application const &A) override;
  virtual void visit(ast::Begin const &B) override;
  virtual void visit(ast::BooleanLiteral const &Bool) override;
  virtual void visit(ast::CaseLambda const &CL) override;
  virtual void visit(ast::CaseLambdaClosure const &CL) override;
  virtual void visit(ast::Char const &C) override;
  virtual void visit(ast::Closure const &L) override;
  virtual void visit(ast::ContinuationMarkSet const &CMS) override;
  virtual void visit(ast::DefineValues const &DV) override;
  virtual void visit(ast::Identifier const &Id) override;
  virtual void visit(ast::IfCond const &If) override;
  virtual void visit(ast::Integer const &Int) override;
  virtual void visit(ast::Keyword const &K) override;
  virtual void visit(ast::Lambda const &L) override;
  virtual void visit(ast::LetValues const &LV) override;
  virtual void visit(ast::Linklet const &Linklet) override;
  virtual void visit(ast::List const &L) override;
  virtual void visit(ast::QuotedExpr const &L) override;
  virtual void visit(ast::RuntimeFunction const &LV) override;
  virtual void visit(ast::SetBang const &SB) override;
  virtual void visit(ast::String const &Str) override;
  virtual void visit(ast::Symbol const &Sym) override;
  virtual void visit(ast::Values const &V) override;
  virtual void visit(ast::VariableReference const &VR) override;
  virtual void visit(ast::Vector const &Vec) override;
  virtual void visit(ast::Void const &Vd) override;
  virtual void visit(ast::WithContinuationMark const &WCM) override;

  // Checks if an identifier is bound in the top-level environment.
  bool isBound(const ast::Identifier &Id) const;

  // Get the current saved result, or null if interpretation failed (e.g. an
  // unbound identifier). main() reports the failure and exits non-zero.
  std::unique_ptr<ast::ValueNode> getResult() const {
    if (!Result) {
      return nullptr;
    }
    return std::unique_ptr<ast::ValueNode>(Result->clone());
  };
  // Peak continuation depth reached across every top-level form run so far.
  // Exposed for the tail-call tests: proper tail calls keep this bounded.
  size_t getPeakKont() const { return PeakKont; }
  std::unique_ptr<ast::ValueNode>
  callFunction(const std::string &Name,
               const llvm::SmallVector<const ast::ValueNode *> &Args) {
    return Runtime::getInstance().callFunction(Name, Args);
  }

private:
  // A continuation frame. Fields are kind-specific; a frame only uses the
  // subset relevant to its kind. Every frame carries a mark map (Marks) and,
  // for frames that later resume evaluating a subexpression, the environment
  // to resume in (Env).
  struct Frame {
    enum Kind {
      Halt,     // bottom of a top-level form's continuation
      Seq,      // begin / begin0 / a multi-expression body
      IfBranch, // choose the then/else branch
      App,      // application: accumulate operator + args, then apply
      MkValues, // (values ...): accumulate then build a Values
      LetBind,  // let-values: accumulate binding values, then bind + body
      LetRec,   // letrec-values: bind each value into the recursive scope
      Define,   // top-level define-values: bind then produce void
      Set,      // set!: mutate then produce void
      WcmKey,   // with-continuation-mark: after key, evaluate val
      WcmVal,   // with-continuation-mark: after val, install mark + eval result
      WcmMark,  // holds a with-continuation-mark mark for its result expression
      Call      // a procedure activation (holds the callee's marks)
    };

    explicit Frame(Kind K) : K(K) {}

    Kind K;
    ast::MarkFrame Marks; // continuation marks belonging to this frame
    EnvPtr Env;           // environment to resume subexpressions in

    // Seq / App / MkValues: the subexpressions to evaluate.
    llvm::SmallVector<const ast::ExprNode *> Exprs;
    size_t Idx = 0; // Seq: index of the next expression to evaluate

    // App / MkValues / LetBind: already-evaluated results (cursor = Done.size).
    std::vector<std::unique_ptr<ast::ValueNode>> Done;

    // Seq (begin0): the saved value of the first expression.
    std::unique_ptr<ast::ValueNode> Saved;
    bool Begin0 = false;

    // IfBranch.
    const ast::ExprNode *ThenE = nullptr;
    const ast::ExprNode *ElseE = nullptr;

    // LetBind / LetRec / Define reference their source node for ids and body.
    const ast::LetValues *Let = nullptr;
    const ast::DefineValues *Def = nullptr;
    EnvPtr DefEnv;   // Define: the scope to define into.
    EnvPtr RecScope; // LetRec: the recursive scope being filled in.

    // App: source location of the application, for arity/procedure errors.
    llvm::SMLoc AppLoc;

    // Set.
    const ast::Identifier *SetId = nullptr;

    // Wcm.
    const ast::ExprNode *WcmValE = nullptr;
    const ast::ExprNode *WcmResultE = nullptr;
    std::unique_ptr<ast::ValueNode> WcmKeyV;

    // Call: owns the applied closure so that Control, which points into the
    // closure's (cloned) lambda body, stays valid for the whole activation.
    std::unique_ptr<ast::ValueNode> Callee;
  };

  enum class Mode { Eval, Continue };

  // Run the machine until the current top-level form's continuation is empty.
  void run();
  // Deliver the value register to the top continuation frame.
  void continueStep();
  // Apply Vals[0] to Vals[1..]. AppLoc/OpLoc anchor arity/procedure errors.
  void applyProcedure(std::vector<std::unique_ptr<ast::ValueNode>> Vals,
                      llvm::SMLoc AppLoc, llvm::SMLoc OpLoc);
  // Evaluate a (non-empty) body sequence in environment E.
  void evalBody(llvm::SmallVector<const ast::ExprNode *> Body, const EnvPtr &E);
  // Create a fresh scope enclosed by Parent, tracked so its bindings can be
  // cleared at teardown to break closure/scope reference cycles.
  EnvPtr newScope(const EnvPtr &Parent);
  // Snapshot the marks on the current continuation, innermost frame first.
  std::vector<ast::MarkFrame> snapshotMarks() const;
  // Set the value register and switch to Continue mode.
  void deliver(std::unique_ptr<ast::ValueNode> V) {
    Val = std::move(V);
    M = Mode::Continue;
  }
  // Abort the current form: unwind to its Halt frame with a null result.
  void abortEval() {
    if (Kont.size() > 1) {
      Kont.erase(Kont.begin() + 1, Kont.end());
    }
    Val = nullptr;
    M = Mode::Continue;
  }

  // Machine state.
  Mode M = Mode::Eval;
  const ast::ASTNode *Control = nullptr;  // expression under evaluation
  EnvPtr Env;                             // current environment
  std::vector<Frame> Kont;                // continuation (top == back())
  std::unique_ptr<ast::ValueNode> Val;    // value register (Continue mode)
  EnvPtr GlobalEnv;                       // top-level scope, persists per form
  std::unique_ptr<ast::ValueNode> Result; // result of the whole linklet
  size_t PeakKont = 0;                    // peak |Kont| seen (tail-call tests)

  // Every scope created during evaluation, so their bindings can be cleared in
  // the destructor. Live-environment closures capture the scope that binds them
  // (top-level lambdas capture GlobalEnv; letrec closures capture the recursive
  // scope), forming shared_ptr cycles that would otherwise never be reclaimed.
  std::vector<EnvPtr> AllScopes;

  nora::DiagnosticEngine &Diag; // diagnostics sink for runtime errors
};
