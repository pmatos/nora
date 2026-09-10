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
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/SMLoc.h>

#include "AST.h"
#include "ASTRuntime.h"
#include "ASTVisitor.h"
#include "Diagnostics.h"
#include "Environment.h"
#include "Runtime.h"
#include "Value.h"
#include "gc_alloc.h"
#include "nora_rt.h"

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
  virtual void visit(ast::Box const &B) override;
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
  virtual void visit(ast::Pair const &P) override;
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
    return std::unique_ptr<ast::ValueNode>(Result.get()->clone());
  };
  // Peak continuation depth reached across every top-level form run so far.
  // Exposed for the tail-call tests: proper tail calls keep this bounded.
  size_t getPeakKont() const { return PeakKont; }
  // Boehm GC live-heap / cumulative-bytes, for the M2 GC forcing seam (a
  // depth-independent live-heap plateau against unbounded churn).
  size_t getGCHeapSize() const { return nrt_gc_heap_size(); }
  size_t getGCTotalBytes() const { return nrt_gc_total_bytes(); }
  std::unique_ptr<ast::ValueNode>
  callFunction(const std::string &Name,
               const llvm::SmallVector<const ast::ValueNode *> &Args) {
    return Runtime::getInstance().callFunction(Name, Args);
  }

private:
  // A continuation frame. Its per-kind transition state is a variant `P` whose
  // active alternative names the kind; each alternative owns only the fields
  // that kind uses, so the compiler enforces "a frame uses only its kind's
  // fields" instead of a comment. Two things are universal, so they live in the
  // header rather than a payload: the mark map (Marks), read by snapshotMarks
  // on every frame and written by setMark on the top activation; and Callee,
  // the applied closure owned by a tail-reusable activation (Call/WcmMark/Halt)
  // so Control, which points into its (cloned) lambda body, outlives the call.
  struct Frame {
    // Halt / WcmMark / Call carry no per-kind state (their content is the
    // header): Halt is a form's bottom, WcmMark holds a with-continuation-mark
    // for its result expression (in Marks), Call is a procedure activation.
    struct Halt {};    // bottom of a top-level form's continuation
    struct WcmMark {}; // holds a with-continuation-mark mark for its result
    struct Call {};    // a procedure activation (holds the callee's marks)

    struct Seq { // begin / begin0 / a multi-expression body
      EnvPtr Env;
      llvm::SmallVector<const ast::ExprNode *> Exprs;
      size_t Idx = 0; // index of the next expression to evaluate
      std::unique_ptr<ast::ValueNode> Saved; // begin0: saved first value
      bool Begin0 = false;
    };
    struct IfBranch { // choose the then/else branch
      EnvPtr Env;
      const ast::ExprNode *ThenE = nullptr;
      const ast::ExprNode *ElseE = nullptr;
    };
    struct App { // application: accumulate operator + args, then apply
      EnvPtr Env;
      llvm::SmallVector<const ast::ExprNode *> Exprs;
      std::vector<std::unique_ptr<ast::ValueNode>> Done; // cursor = Done.size()
      llvm::SMLoc AppLoc; // source location, for arity/procedure errors
    };
    struct MkValues { // (values ...): accumulate then build a Values
      EnvPtr Env;
      llvm::SmallVector<const ast::ExprNode *> Exprs;
      std::vector<std::unique_ptr<ast::ValueNode>> Done;
    };
    struct LetBind { // let-values: accumulate binding values, then bind + body
      EnvPtr Env;
      const ast::LetValues *Let = nullptr;
      std::vector<std::unique_ptr<ast::ValueNode>> Done;
    };
    struct LetRec { // letrec-values: bind each value into the recursive scope
      const ast::LetValues *Let = nullptr;
      EnvPtr RecScope; // the recursive scope being filled in
      size_t Idx = 0;
    };
    struct Define { // top-level define-values: bind then produce void
      const ast::DefineValues *Def = nullptr;
      EnvPtr DefEnv; // the scope to define into
    };
    struct Set { // set!: mutate then produce void
      EnvPtr Env;
      const ast::Identifier *SetId = nullptr;
    };
    struct WcmKey { // with-continuation-mark: after key, evaluate val
      EnvPtr Env;
      const ast::ExprNode *WcmValE = nullptr;
      const ast::ExprNode *WcmResultE = nullptr;
    };
    struct WcmVal { // with-continuation-mark: after val, install mark + result
      EnvPtr Env;
      const ast::ExprNode *WcmResultE = nullptr;
      std::unique_ptr<ast::ValueNode> WcmKeyV;
    };

    // The three tail-reusable activation kinds (Call/WcmMark/Halt) are the last
    // three alternatives so isReusable()/isHalt() are unsigned index compares;
    // the static_asserts below pin that ordering, so a new kind inserted before
    // them leaves both predicates correct with no constant to edit.
    using Payload =
        std::variant<Seq, IfBranch, App, MkValues, LetBind, LetRec, Define, Set,
                     WcmKey, WcmVal, Halt, WcmMark, Call>;

    ast::MarkFrame Marks;                   // marks belonging to this frame
    std::unique_ptr<ast::ValueNode> Callee; // activation's owned closure
    Payload P;

    // Construct from any payload alternative; excludes Frame itself so the
    // implicit move constructor (used by vector growth) is not hijacked.
    template <typename Alt, typename = std::enable_if_t<
                                !std::is_same_v<std::decay_t<Alt>, Frame>>>
    explicit Frame(Alt &&A) : P(std::forward<Alt>(A)) {}

    static constexpr size_t HaltIdx = std::variant_size_v<Payload> - 3;
    bool isHalt() const { return P.index() == HaltIdx; }
    bool isReusable() const { return P.index() >= HaltIdx; }
  };
  static_assert(std::variant_size_v<Frame::Payload> == 13);
  static_assert(
      std::is_same_v<std::variant_alternative_t<Frame::HaltIdx, Frame::Payload>,
                     Frame::Halt>);
  static_assert(std::is_same_v<
                std::variant_alternative_t<Frame::HaltIdx + 1, Frame::Payload>,
                Frame::WcmMark>);
  static_assert(std::is_same_v<
                std::variant_alternative_t<Frame::HaltIdx + 2, Frame::Payload>,
                Frame::Call>);

  // Push a fresh continuation frame carrying payload Init, and return a
  // reference to the just-constructed payload so the caller can fill its
  // fields.
  template <typename K> K &pushK(K Init) {
    Kont.emplace_back(Frame(std::move(Init)));
    return std::get<K>(Kont.back().P);
  }

  // Continue-mode transitions: deliver the value register to the top frame.
  // One overload per kind (the former continueStep switch arms); dispatched by
  // std::visit. step(Halt) is unreachable - run() intercepts Halt.
  void step(Frame::Seq &K);
  void step(Frame::IfBranch &K);
  void step(Frame::App &K);
  void step(Frame::MkValues &K);
  void step(Frame::LetBind &K);
  void step(Frame::LetRec &K);
  void step(Frame::Define &K);
  void step(Frame::Set &K);
  void step(Frame::WcmKey &K);
  void step(Frame::WcmVal &K);
  void step(Frame::WcmMark &K);
  void step(Frame::Call &K);
  void step(Frame::Halt &K);

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
  void deliver(Value V) {
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
  const ast::ASTNode *Control = nullptr; // expression under evaluation
  EnvPtr Env;                            // current environment
  // The continuation. Its backing store is GC-allocated (and scanned) so that,
  // as values migrate onto the GC heap, in-flight values held in frames stay
  // reachable — the Kont header lives in this stack-resident Interpreter, so
  // Boehm's stack scan roots the buffer. (M2/GC S1.)
  std::vector<Frame, GcAllocator<Frame>> Kont; // continuation (top == back())
  Value Val;                                   // value register (Continue mode)
  EnvPtr GlobalEnv;    // top-level scope, persists per form
  Value Result;        // result of the whole linklet
  size_t PeakKont = 0; // peak |Kont| seen (tail-call tests)

  // Every scope created during evaluation, so their bindings can be cleared in
  // the destructor. Live-environment closures capture the scope that binds them
  // (top-level lambdas capture GlobalEnv; letrec closures capture the recursive
  // scope), forming shared_ptr cycles that would otherwise never be reclaimed.
  std::vector<EnvPtr> AllScopes;

  nora::DiagnosticEngine &Diag; // diagnostics sink for runtime errors
};
