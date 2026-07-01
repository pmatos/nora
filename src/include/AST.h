#pragma once

#include "ASTVisitor.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <compare>
#include <gmp.h>
#include <iostream>
#include <memory>
#include <ranges>
#include <utility>

// Forward declarations
class IdPool;

namespace ast {

// Forward definition
class ValueNode;

// Definition of top-level abstract nodes
class ASTNode {
public:
  // The order of the enum is important. The order of the nodes in the enum
  // must match the order of the nodes in the class hierarchy. The order of
  // the nodes in the class hierarchy are in lexical order.
  // Note: Keep the following list in sorted order
  enum ASTNodeKind {
    AST_Linklet,
    First_TLNode, // all TLNodes must be after this
    AST_DefineValues,
    First_ExprNode, // all ExprNodes must be after this
    AST_Application,
    AST_Begin,
    AST_Identifier,
    AST_IfCond,
    AST_LetValues,
    AST_SetBang,
    First_ValueNode, // all ValueNodes must be after this
    AST_BooleanLiteral,
    AST_CaseLambda,
    AST_CaseLambdaClosure, // result of evaluating a CaseLambda expression
    AST_Char,
    AST_Closure, // result of evaluating a Lambda expression
    AST_Integer,
    AST_Keyword,
    AST_Lambda,
    AST_List,
    AST_String,
    AST_Symbol,
    AST_Values,
    AST_Vector,
    AST_Void,
    AST_RuntimeFunction,
    AST_QuotedExpr // result of a quote expression
  };

  ASTNode(ASTNodeKind Kind) : Kind(Kind) {}
  virtual ~ASTNode() = default;

  [[nodiscard]] ASTNodeKind getKind() const { return Kind; }
  virtual void dump() const = 0;
  virtual ASTNode *clone() const = 0;
  virtual void accept(ASTVisitor &Visitor) const = 0;

private:
  const ASTNodeKind Kind;
};

class TLNode : public ASTNode {
public:
  TLNode(ASTNodeKind Kind) : ASTNode(Kind) {}

  virtual TLNode *clone() const override = 0;
  virtual void dump() const override = 0;

  static bool classof(const ASTNode *N) {
    return N->getKind() > ASTNodeKind::First_TLNode;
  }
};

class ExprNode : public TLNode {
public:
  ExprNode(ASTNodeKind Kind) : TLNode(Kind) {}

  virtual ExprNode *clone() const override = 0;
  virtual void dump() const override = 0;

  static bool classof(const ASTNode *N) {
    return N->getKind() > ASTNodeKind::First_ExprNode;
  }
};

class ValueNode : public ExprNode {
public:
  ValueNode(ASTNodeKind Kind) : ExprNode(Kind) {}

  virtual ValueNode *clone() const override = 0;
  virtual void dump() const override = 0;
  virtual void write() const = 0;

  static bool classof(const ASTNode *N) {
    return N->getKind() > ASTNodeKind::First_ValueNode;
  }
};

// Template class to ease the definition of method clone and accept
// on leaf node classes
template <typename Derived, typename Base> class ClonableNode : public Base {
public:
  ClonableNode(ASTNode::ASTNodeKind Kind) : Base(Kind) {}

  Base *clone() const override {
    return new Derived(static_cast<const Derived &>(*this));
  }
  void accept(ASTVisitor &Visitor) const override {
    Visitor.visit(static_cast<const Derived &>(*this));
  }
};

// Leaf Nodes
class Identifier : public ClonableNode<Identifier, ExprNode> {
public:
  Identifier(const Identifier &I)
      : ClonableNode(ASTNodeKind::AST_Identifier), Id(I.Id) {}
  Identifier(Identifier &&) = default;
  Identifier &operator=(const Identifier &I) {
    Id = I.Id;
    return *this;
  }
  Identifier &operator=(Identifier &&I) noexcept;
  virtual ~Identifier() = default;

  std::strong_ordering operator<=>(const Identifier &I) const;

  [[nodiscard]] const llvm::StringRef getName() const;
  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Identifier;
  }

private:
  friend class ::IdPool;
  explicit Identifier(llvm::StringRef Id);
  llvm::StringRef Id;
};

// A string datum, e.g. "z". Value holds the string's read lexeme including the
// surrounding double quotes, which is also its Racket `print` form for the
// escapes we currently support. Strings are self-quoting.
class String : public ClonableNode<String, ValueNode> {
public:
  explicit String(llvm::StringRef Value)
      : ClonableNode(ASTNodeKind::AST_String), Value(Value) {}
  String(const String &S)
      : ClonableNode(ASTNodeKind::AST_String), Value(S.Value) {}
  String(String &&) = default;
  ~String() = default;

  [[nodiscard]] llvm::StringRef getValue() const { return Value; }
  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_String;
  }

private:
  llvm::SmallString<16> Value;
};

class Symbol : public ClonableNode<Symbol, ValueNode> {
public:
  explicit Symbol(llvm::StringRef Name)
      : ClonableNode(ASTNodeKind::AST_Symbol), Name(Name) {}
  Symbol(const Symbol &S)
      : ClonableNode(ASTNodeKind::AST_Symbol), Name(S.Name) {}
  Symbol(Symbol &&) = default;
  Symbol &operator=(const Symbol &S) = delete;
  Symbol &operator=(Symbol &&S) = delete;
  virtual ~Symbol() = default;

  // FIXME: symbols are not interned yet, so eq?/eqv? identity is approximated
  // by comparing names. Interning is future work.
  bool operator==(const Symbol &S) const { return getName() == S.getName(); }

  [[nodiscard]] llvm::StringRef getName() const { return Name; }
  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Symbol;
  }

private:
  llvm::SmallString<32> Name;
};

// A keyword datum, e.g. #:foo. Name holds the bare keyword (without the leading
// #:), matching the lexer's KEYWORD token. Like symbols, keywords are not
// self-quoting, so the enclosing QuotedExpr emits a leading quote ('#:foo).
class Keyword : public ClonableNode<Keyword, ValueNode> {
public:
  explicit Keyword(llvm::StringRef Name)
      : ClonableNode(ASTNodeKind::AST_Keyword), Name(Name) {}
  Keyword(const Keyword &K)
      : ClonableNode(ASTNodeKind::AST_Keyword), Name(K.Name) {}
  Keyword(Keyword &&) = default;
  Keyword &operator=(const Keyword &K) = delete;
  Keyword &operator=(Keyword &&K) = delete;
  virtual ~Keyword() = default;

  bool operator==(const Keyword &K) const { return getName() == K.getName(); }

  [[nodiscard]] llvm::StringRef getName() const { return Name; }
  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Keyword;
  }

private:
  llvm::SmallString<32> Name;
};

class Linklet : public ClonableNode<Linklet, ASTNode> {
public:
  // helper type
  using idpair_t = std::pair<Identifier, Identifier>;

  Linklet() : ClonableNode(ASTNodeKind::AST_Linklet) {}
  Linklet(const Linklet &L);
  Linklet(Linklet &&L)
      : ClonableNode(ASTNodeKind::AST_Linklet), Imports(std::move(L.Imports)),
        Exports(std::move(L.Exports)), Body(std::move(L.Body)) {}
  Linklet &operator=(const Linklet &L) = delete;
  ~Linklet() = default;

  class FormRange {
  public:
    FormRange() = delete;
    FormRange(
        llvm::SmallVector<std::unique_ptr<TLNode>>::const_iterator FsBegin,
        llvm::SmallVector<std::unique_ptr<TLNode>>::const_iterator FsEnd);
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    [[nodiscard]] TLNode const &operator[](size_t I) const;

  private:
    llvm::SmallVector<std::unique_ptr<TLNode>>::const_iterator BeginIt, EndIt;
  };

  void appendImport(const Identifier &ExtId, const Identifier &IntId);
  void appendExport(const Identifier &IntId, const Identifier &ExtId);
  void appendBodyForm(std::unique_ptr<TLNode> &&Form);

  size_t exportsCount() const { return Exports.size(); }
  size_t importsCount() const { return Imports.size(); }
  size_t bodyFormsCount() const { return Body.size(); }

  [[nodiscard]] FormRange getBody() const {
    return {Body.begin(), Body.end()};
  };
  [[nodiscard]] const llvm::SmallVector<idpair_t> &getImports() const {
    return Imports;
  }
  [[nodiscard]] const llvm::SmallVector<idpair_t> &getExports() const {
    return Exports;
  }

  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Linklet;
  }

private:
  // Sets of imports. The pair contains two Ids
  // One for external-imported-id and the other for internal-imported-id
  llvm::SmallVector<idpair_t> Imports;

  // Sets of exports. The pair contains two Ids
  // One for internal-exported-id and the other for external-exported-id
  // If there is just a single exported-id, then both are the same
  llvm::SmallVector<idpair_t> Exports;

  /// Linklet body expressions
  llvm::SmallVector<std::unique_ptr<TLNode>> Body;
};

class Application : public ClonableNode<Application, ExprNode> {
public:
  Application() : ClonableNode(ASTNodeKind::AST_Application) {}
  Application &operator=(Application &&) = delete;
  Application(const Application &);
  Application(Application &&) = default;
  Application &operator=(const Application &) = delete;
  ~Application() = default;

  void appendExpr(std::unique_ptr<ExprNode> &&Expr);
  [[nodiscard]] size_t length() const { return Exprs.size(); }
  [[nodiscard]] ExprNode const &operator[](size_t I) const;

  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Application;
  }

  const llvm::SmallVector<std::unique_ptr<ExprNode>> &getExprs() const {
    return Exprs;
  }

private:
  llvm::SmallVector<std::unique_ptr<ExprNode>> Exprs;
};

// AST Node representing a begin or begin0 expression.
class Begin : public ClonableNode<Begin, ExprNode> {
public:
  Begin() : ClonableNode(ASTNodeKind::AST_Begin) {}
  Begin(const Begin &B);
  Begin(Begin &&B) = default;
  Begin &operator=(const Begin &B) = delete;
  ~Begin() = default;

  [[nodiscard]] const llvm::SmallVector<std::unique_ptr<ExprNode>> &
  getBody() const {
    return Body;
  }
  [[nodiscard]] size_t bodyCount() const { return Body.size(); }
  [[nodiscard]] bool isZero() const { return Zero; }
  LLVM_DUMP_METHOD void dump() const override {}

  void appendExpr(std::unique_ptr<ExprNode> &&E);
  void markAsBegin0() { Zero = true; }

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Begin;
  }

private:
  llvm::SmallVector<std::unique_ptr<ExprNode>> Body;
  bool Zero = false;
};

class BooleanLiteral : public ClonableNode<BooleanLiteral, ValueNode> {
public:
  BooleanLiteral() : ClonableNode(ASTNodeKind::AST_BooleanLiteral) {}
  BooleanLiteral(bool V)
      : ClonableNode(ASTNodeKind::AST_BooleanLiteral), Value(V) {}
  BooleanLiteral(const BooleanLiteral &) = default;
  BooleanLiteral(BooleanLiteral &&) = default;
  ~BooleanLiteral() = default;

  [[nodiscard]] bool value() const { return Value; }
  LLVM_DUMP_METHOD void dump() const override {
    if (Value)
      llvm::dbgs() << "#t";
    else
      llvm::dbgs() << "#f";
  }
  // Use std::cout, like the other value writers, so booleans interleave in
  // order with the surrounding quote/list/vector output (llvm::outs() buffers
  // independently and would reorder e.g. '(#t 1 #f 2)).
  void write() const override { std::cout << (Value ? "#t" : "#f"); }

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_BooleanLiteral;
  }

private:
  bool Value;
};

// A character datum, e.g. #\a. Value holds the character's printed glyph(s)
// without the leading #\ (so "a" for #\a). Characters are self-quoting.
class Char : public ClonableNode<Char, ValueNode> {
public:
  explicit Char(llvm::StringRef Value)
      : ClonableNode(ASTNodeKind::AST_Char), Value(Value) {}
  Char(const Char &C) : ClonableNode(ASTNodeKind::AST_Char), Value(C.Value) {}
  Char(Char &&) = default;
  ~Char() = default;

  [[nodiscard]] llvm::StringRef getValue() const { return Value; }
  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Char;
  }

private:
  llvm::SmallString<8> Value;
};

class DefineValues : public ClonableNode<DefineValues, TLNode> {
public:
  DefineValues(llvm::SmallVector<Identifier> Ids,
               std::unique_ptr<ExprNode> &Body);
  DefineValues(const DefineValues &DV);
  DefineValues(DefineValues &&DV) = default;
  ~DefineValues() = default;

  // View over the Ids
  // FIXME: we should be able to use C++20 view_interface here
  // although my initial attempt failed.
  class IdRange {
  public:
    IdRange() = delete;
    IdRange(const llvm::SmallVector<Identifier> &Ids)
        : BeginIt(Ids.begin()), EndIt(Ids.end()) {}
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    const Identifier &operator[](size_t Idx) const { return *(BeginIt + Idx); }

  private:
    llvm::SmallVector<Identifier>::const_iterator BeginIt, EndIt;
  };

  IdRange getIds() const { return IdRange{Ids}; }
  [[nodiscard]] const ExprNode &getBody() const;
  [[nodiscard]] size_t countIds() const { return Ids.size(); }
  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_DefineValues;
  }

private:
  llvm::SmallVector<Identifier> Ids;
  std::unique_ptr<ExprNode> Body;
};

// There are three types of lambda formals:
// 1. A formal that is an identifier. This is when we want to pass a single
//   list into the lambda.
//   (lambda x (do-something x)) , and I can pass as many values as I want and
//   they are all aggregated into a list.
// 2. A formal that is a list of identifiers. This is when we want to pass a
// specific
//    number of arguments into the lambda and each argumend is bound to a
//    formal.
// 3. A formal that is a list of identifiers and a rest identifier. This is
// when we want to
//    to pass a minimum number of arguments into the lambda. Each argument up
//    to the number of formals is bound to a formal. The rest of the arguments
//    are bound to the rest identifier as a list.
class Formal {
public:
  enum class Type { Identifier, List, ListRest };
  virtual ~Formal() = default;
  virtual std::unique_ptr<Formal> clone() const = 0;

  virtual Type getType() const = 0;
};

class IdentifierFormal : public Formal {
public:
  IdentifierFormal(const Identifier &I) : I(I) {};
  IdentifierFormal(const IdentifierFormal &Other) = default;
  IdentifierFormal(IdentifierFormal &&Other) = default;
  IdentifierFormal &operator=(const IdentifierFormal &Other) = delete;
  IdentifierFormal &operator=(IdentifierFormal &&Other) = default;
  ~IdentifierFormal() = default;

  std::unique_ptr<Formal> clone() const override {
    return std::make_unique<IdentifierFormal>(*this);
  }

  [[nodiscard]] Type getType() const override { return Type::Identifier; }
  [[nodiscard]] const Identifier &getIdentifier() const { return I; }

private:
  Identifier I;
};

class ListFormal : public Formal {
public:
  ListFormal() = default;
  ListFormal(const llvm::SmallVector<Identifier> &Ids) : Formals(Ids) {}
  ListFormal(ListFormal const &Other) = default;
  ListFormal(ListFormal &&Other) = default;
  ListFormal &operator=(const ListFormal &Other) = delete;
  ListFormal &operator=(ListFormal &&Other) = default;
  ~ListFormal() override = default;

  std::unique_ptr<Formal> clone() const override {
    return std::make_unique<ListFormal>(*this);
  }

  class IdRange {
  public:
    IdRange() = delete;
    IdRange(const llvm::SmallVector<Identifier> &Ids)
        : BeginIt(Ids.begin()), EndIt(Ids.end()) {}
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }

  private:
    llvm::SmallVector<Identifier>::const_iterator BeginIt, EndIt;
  };

  IdRange getIds() const { return IdRange{Formals}; }
  [[nodiscard]] size_t size() const { return Formals.size(); }
  [[nodiscard]] Type getType() const override { return Type::List; }
  void addFormal(const Identifier &I);
  Identifier &operator[](size_t Index) { return Formals[Index]; }

private:
  llvm::SmallVector<Identifier> Formals;
};

class ListRestFormal : public ListFormal {
public:
  ListRestFormal(const llvm::SmallVector<Identifier> &Formals,
                 const Identifier &RestFormal)
      : ListFormal(Formals), RestFormal(RestFormal) {}
  ListRestFormal(const ListRestFormal &Other) = default;
  ListRestFormal(ListRestFormal &&Other) = default;
  ListRestFormal &operator=(const ListRestFormal &Other) = delete;
  ListRestFormal &operator=(ListRestFormal &&Other) = default;
  ~ListRestFormal() override = default;

  std::unique_ptr<Formal> clone() const override {
    return std::make_unique<ListRestFormal>(*this);
  }
  [[nodiscard]] Type getType() const override { return Type::ListRest; }
  void addRestFormal(const Identifier &I);
  const Identifier &getRestFormal() const { return RestFormal; }

private:
  Identifier RestFormal;
};

class IfCond : public ClonableNode<IfCond, ExprNode> {
public:
  IfCond() : ClonableNode(ASTNodeKind::AST_IfCond) {}
  IfCond(const IfCond &);
  IfCond(IfCond &&) = default;
  ~IfCond() = default;

  void setCond(std::unique_ptr<ExprNode> &&Cond);
  void setThen(std::unique_ptr<ExprNode> &&Then);
  void setElse(std::unique_ptr<ExprNode> &&Else);

  [[nodiscard]] ExprNode const &getCond() const;
  [[nodiscard]] ExprNode const &getThen() const;
  [[nodiscard]] ExprNode const &getElse() const;

  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_IfCond;
  }

private:
  std::unique_ptr<ExprNode> Cond;
  std::unique_ptr<ExprNode> Then;
  std::unique_ptr<ExprNode> Else;
};

class Integer : public ClonableNode<Integer, ValueNode> {
public:
  explicit Integer(const std::string &V);
  explicit Integer(int64_t V);
  Integer(const Integer &I);
  Integer(Integer &&Int) = default;
  Integer &operator=(const Integer &Int);
  ~Integer();

  Integer &operator+=(const Integer &Int);
  Integer &operator-=(const Integer &Int);
  Integer &operator*=(const Integer &Int);

  bool operator==(int64_t X) const;
  bool operator==(const Integer &Int) const;

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;
  std::string asString() const;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Integer;
  }

private:
  mpz_t Value;
};

class Lambda : public ClonableNode<Lambda, ValueNode> {
public:
  Lambda() : ClonableNode(ASTNodeKind::AST_Lambda) {}
  Lambda(Lambda const &L);
  Lambda(Lambda &&L) = default;
  ~Lambda() = default;

  void setFormals(std::unique_ptr<Formal> F) { Formals = std::move(F); }
  void setBody(std::unique_ptr<ExprNode> E) { Body = std::move(E); }

  Formal::Type getFormalsType() const { return Formals->getType(); }

  [[nodiscard]] Formal const &getFormals() const { return *Formals; }
  [[nodiscard]] const ExprNode &getBody() const;

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  llvm::SmallVector<Identifier> findFreeVariables() const;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Lambda;
  }

private:
  std::unique_ptr<Formal> Formals;
  std::unique_ptr<ExprNode> Body;
};

// A case-lambda is a sequence of lambda clauses. When applied, the first
// clause whose formals accept the number of supplied arguments is selected.
// Defined in terms of Lambda, so it is placed right after it.
class CaseLambda : public ClonableNode<CaseLambda, ValueNode> {
public:
  CaseLambda() : ClonableNode(ASTNodeKind::AST_CaseLambda) {}
  CaseLambda(CaseLambda const &CL);
  CaseLambda(CaseLambda &&CL) = default;
  ~CaseLambda() = default;

  void addClause(std::unique_ptr<Lambda> C) { Clauses.push_back(std::move(C)); }
  [[nodiscard]] size_t size() const { return Clauses.size(); }
  [[nodiscard]] Lambda const &operator[](size_t Idx) const {
    return *Clauses[Idx];
  }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_CaseLambda;
  }

private:
  llvm::SmallVector<std::unique_ptr<Lambda>> Clauses;
};

class LetValues : public ClonableNode<LetValues, ExprNode> {
public:
  LetValues() : ClonableNode(ASTNodeKind::AST_LetValues) {}
  LetValues &operator=(const LetValues &) = delete;
  LetValues &operator=(LetValues &&) = delete;
  LetValues(const LetValues &DV);
  LetValues(LetValues &&DV) = default;
  ~LetValues() = default;

  // View over the Ids
  // FIXME: we should be able to use C++20 view_interface here
  // although my initial attempt failed.
  class IdRange {
  public:
    IdRange() = delete;
    IdRange(const llvm::SmallVector<Identifier> &Ids)
        : BeginIt(Ids.begin()), EndIt(Ids.end()) {}
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    [[nodiscard]] Identifier const &operator[](size_t Idx) const {
      return *(BeginIt + Idx);
    }

  private:
    llvm::SmallVector<Identifier>::const_iterator BeginIt, EndIt;
  };

  IdRange getBindingIds(size_t Idx) const;
  ExprNode const &getBindingExpr(size_t Idx) const;
  ExprNode const &getBodyExpr(size_t Idx) const;

  void appendBinding(llvm::SmallVector<Identifier> &&Ids,
                     std::unique_ptr<ExprNode> Expr);
  void appendBody(std::unique_ptr<ExprNode> Expr);

  size_t bindingCount() const {
    assert(Ids.size() == Exprs.size());
    return Ids.size();
  }
  size_t idsCount() const { return Ids.size(); }
  size_t exprsCount() const;
  size_t bodyCount() const { return Body.size(); }

  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_LetValues;
  }

private:
  llvm::SmallVector<llvm::SmallVector<Identifier>> Ids;
  llvm::SmallVector<std::unique_ptr<ExprNode>> Exprs;
  llvm::SmallVector<std::unique_ptr<ExprNode>> Body;
};

class List : public ClonableNode<List, ValueNode> {
public:
  List() : ClonableNode(ASTNodeKind::AST_List) {}
  List(List const &L);
  List(List &&L) = default;
  ~List() = default;

  void appendExpr(std::unique_ptr<ValueNode> &&Expr);
  [[nodiscard]] size_t length() const { return Values.size(); }
  [[nodiscard]] ValueNode const &operator[](size_t I) const;

  // A dotted (improper) list carries a non-list tail after the last element,
  // e.g. (1 2 . the-end). A null tail denotes a proper list.
  void setTail(std::unique_ptr<ValueNode> &&T) { Tail = std::move(T); }
  [[nodiscard]] bool hasTail() const { return Tail != nullptr; }
  [[nodiscard]] const ValueNode *getTail() const { return Tail.get(); }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_List;
  }

  auto const &values() const { return Values; }

private:
  llvm::SmallVector<std::unique_ptr<ast::ValueNode>> Values;
  std::unique_ptr<ast::ValueNode> Tail;
};

class SetBang : public ClonableNode<SetBang, ExprNode> {
public:
  SetBang() : ClonableNode(ASTNodeKind::AST_SetBang) {}
  SetBang(const SetBang &);
  SetBang(SetBang &&) = default;
  ~SetBang() = default;

  void setIdentifier(std::unique_ptr<Identifier> &&Id);
  void setExpr(std::unique_ptr<ExprNode> &&Expr);

  [[nodiscard]] Identifier const &getIdentifier() const;
  [[nodiscard]] ExprNode const &getExpr() const;

  LLVM_DUMP_METHOD void dump() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_SetBang;
  }

private:
  std::unique_ptr<Identifier> Id;
  std::unique_ptr<ExprNode> Expr;
};

class Values : public ClonableNode<Values, ValueNode> {
public:
  explicit Values(llvm::SmallVector<std::unique_ptr<ExprNode>> Exprs);
  Values(const Values &V);
  Values(Values &&V) = default;
  Values &operator=(const Values &V) = delete;

  ~Values() = default;

  class ExprRange {
  public:
    ExprRange() = delete;
    ExprRange(
        llvm::SmallVector<std::unique_ptr<ExprNode>>::const_iterator EsBegin,
        llvm::SmallVector<std::unique_ptr<ExprNode>>::const_iterator EsEnd);
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    [[nodiscard]] ExprNode const &operator[](size_t I) const;

  private:
    llvm::SmallVector<std::unique_ptr<ExprNode>>::const_iterator BeginIt, EndIt;
  };

  [[nodiscard]] ExprRange getExprs() const {
    return {Exprs.begin(), Exprs.end()};
  }
  [[nodiscard]] size_t countExprs() const { return Exprs.size(); }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Values;
  }

private:
  llvm::SmallVector<std::unique_ptr<ExprNode>> Exprs;
};

// A vector datum, e.g. #(1 2). Unlike lists, vectors are NOT self-quoting, so
// at the top level QuotedExpr prints a leading quote ('#(1 2)); nested inside
// other data they print in bare read syntax (#(1 2)).
class Vector : public ClonableNode<Vector, ValueNode> {
public:
  Vector() : ClonableNode(ASTNodeKind::AST_Vector) {}
  Vector(Vector const &V);
  Vector(Vector &&V) = default;
  ~Vector() = default;

  void appendExpr(std::unique_ptr<ValueNode> &&Expr);
  [[nodiscard]] size_t length() const { return Values.size(); }
  [[nodiscard]] ValueNode const &operator[](size_t I) const;

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Vector;
  }

  auto const &values() const { return Values; }

private:
  llvm::SmallVector<std::unique_ptr<ast::ValueNode>> Values;
};

// This class represents the void constant.
// https://docs.racket-lang.org/guide/void_undefined.html
class Void : public ClonableNode<Void, ValueNode> {
public:
  Void() : ClonableNode(ASTNodeKind::AST_Void) {}
  Void(const Void &V) = default;
  Void(Void &&V) = default;
  Void &operator=(const Void &V) = delete;
  bool operator==(const Void &V) const { return true; }
  ~Void() = default;

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_Void;
  }

private:
};

// This class represents the void constant.
// https://docs.racket-lang.org/reference/quote.html
class QuotedExpr : public ClonableNode<QuotedExpr, ValueNode> {
public:
  QuotedExpr();
  QuotedExpr(const QuotedExpr &QE);
  QuotedExpr(QuotedExpr &&V) = default;
  QuotedExpr &operator=(const QuotedExpr &V) = delete;
  bool operator==(const QuotedExpr &V) const;
  ~QuotedExpr() = default;

  void setQuotedExpr(std::unique_ptr<ValueNode> &&Expr);
  [[nodiscard]] const ValueNode &getQuotedExpr() const { return *Node; }

  LLVM_DUMP_METHOD void dump() const override;
  void write() const override;

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_QuotedExpr;
  }

private:
  std::unique_ptr<ValueNode> Node;
};

class RuntimeFunction : public ValueNode {
public:
  RuntimeFunction(const std::string &Name)
      : ValueNode(ASTNodeKind::AST_RuntimeFunction), Name(Name) {}

  virtual ~RuntimeFunction() = default;
  virtual std::unique_ptr<ast::ValueNode>
  operator()(const llvm::SmallVector<const ast::ValueNode *> &Args) const = 0;

  LLVM_DUMP_METHOD void dump() const override {
    llvm::dbgs() << "#<runtime:" << getName() << ">";
  }
  void write() const override {
    llvm::outs() << "#<runtime:" << getName() << ">";
  }

  static bool classof(const ASTNode *N) {
    return N->getKind() == ASTNodeKind::AST_RuntimeFunction;
  }

  virtual RuntimeFunction *clone() const override = 0;

  const std::string &getName() const { return Name; }

private:
  std::string Name;
};

}; // namespace ast