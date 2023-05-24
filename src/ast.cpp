#include "ast.h"

#include <llvm/Support/Casting.h>
#include <plog/Log.h>

using namespace ast;

//
// Implementation of Application node.
//

// Copy Constructor for Application.
Application::Application(const Application &A)
    : ClonableNode(ASTNodeKind::AST_Application) {
  Exprs.reserve(A.Exprs.size());
  for (const auto &Expr : A.Exprs) {
    Exprs.emplace_back(Expr->clone());
  }
}

void Application::appendExpr(std::unique_ptr<ExprNode> &&E) {
  Exprs.emplace_back(std::move(E));
}

ExprNode const &Application::operator[](size_t I) const { return *Exprs[I]; }

void Application::dump() const {
  std::wcout << L"(";
  for (const auto &Expr : Exprs) {
    Expr->dump();
    std::wcout << L" ";
  }
  std::wcout << L")";
}

//
// Implementation of Begin node.
//

// Copy Constructor for Begin.
Begin::Begin(const Begin &B) : ClonableNode(ASTNodeKind::AST_Begin) {
  for (const auto &Expr : B.Body) {
    std::unique_ptr<ExprNode> Ptr(Expr->clone());
    Body.emplace_back(std::move(Ptr));
  }
}

void Begin::appendExpr(std::unique_ptr<ExprNode> &&E) {
  Body.emplace_back(std::move(E));
}

//
// Implementation of Identifier node.
//

Identifier::Identifier(std::wstring_view Id)
    : ClonableNode(ASTNodeKind::AST_Identifier), Id(Id) {}

Identifier &Identifier::operator=(Identifier &&I) noexcept {
  Id = I.Id;
  return *this;
}

const std::wstring_view Identifier::getName() const { return Id; }

std::strong_ordering Identifier::operator<=>(const Identifier &I) const {
  return Id <=> I.Id;
}

void Identifier::dump() const {
  // FIXME: use ICU for proper output to llvm::dbgs()
  std::wcout << Id;
}

//
// Implementation of IfCond node.
//

// Copy Constructor for IfCond.
IfCond::IfCond(const IfCond &I)
    : ClonableNode(ASTNodeKind::AST_IfCond), Cond(I.Cond->clone()),
      Then(I.Then->clone()), Else(I.Else->clone()) {}

void IfCond::setCond(std::unique_ptr<ExprNode> &&C) { Cond = std::move(C); }
void IfCond::setThen(std::unique_ptr<ExprNode> &&T) { Then = std::move(T); }
void IfCond::setElse(std::unique_ptr<ExprNode> &&E) { Else = std::move(E); }

ExprNode const &IfCond::getCond() const { return *Cond; }
ExprNode const &IfCond::getThen() const { return *Then; }
ExprNode const &IfCond::getElse() const { return *Else; }

void IfCond::dump() const {
  std::wcout << L"(if ";
  Cond->dump();
  std::wcout << L" ";
  Then->dump();
  std::wcout << L" ";
  Else->dump();
  std::wcout << L")";
}

//
// Implementation of Integer node.
//

Integer::Integer(const std::string &V)
    : ClonableNode(ASTNodeKind::AST_Integer) {
  mpz_init_set_str(Value, V.c_str(), 10);
}

Integer::Integer(int64_t V) : ClonableNode(ASTNodeKind::AST_Integer) {
  mpz_init_set_si(Value, V);
}

Integer::Integer(const Integer &I) : ClonableNode(ASTNodeKind::AST_Integer) {
  mpz_init_set(Value, I.Value);
}

Integer::~Integer() { mpz_clear(Value); }

Integer &Integer::operator=(const Integer &Int) {
  mpz_set(Value, Int.Value);
  return *this;
}

bool Integer::operator==(int64_t X) const { return mpz_cmp_si(Value, X) == 0; }
bool Integer::operator==(const Integer &Int) const {
  return mpz_cmp(Value, Int.Value) == 0;
}
Integer &Integer::operator+=(const Integer &Int) {
  mpz_add(Value, Value, Int.Value);
  return *this;
}

Integer &Integer::operator-=(const Integer &Int) {
  mpz_sub(Value, Value, Int.Value);
  return *this;
}

Integer &Integer::operator*=(const Integer &Int) {
  mpz_mul(Value, Value, Int.Value);
  return *this;
}

void Integer::dump() const {
  // FIXME: print to err using llvm::dbgs()
  // FIXME: why doesn't this work properly ? gmp_fprintf(stderr, "%Zd", Value);
  std::cerr << asString();
}

void Integer::write() const { gmp_printf("%Zd", Value); }

std::string Integer::asString() const {
  char *Str = mpz_get_str(nullptr, 10, Value);
  std::string S(Str);
  free(Str);
  return S;
}

//
// Implementation of Lambda node.
//

// Copy Constructor for Lambda.
Lambda::Lambda(Lambda const &L)
    : ClonableNode(ASTNodeKind::AST_Lambda), Body(L.Body->clone()) {
  Formals = L.Formals->clone();
}

const ExprNode &Lambda::getBody() const { return *Body; }

void Lambda::dump() const {
  std::wcout << L"(lambda ";
  auto const &F = getFormals();
  switch (F.getType()) {
  case Formal::Type::Identifier: {
    IdentifierFormal const &FId = static_cast<IdentifierFormal const &>(F);
    FId.getIdentifier().dump();
    break;
  }
  case Formal::Type::List: {
    ListFormal const &FList = static_cast<ListFormal const &>(F);
    std::cout << "(";
    for (const auto &Formal : FList.getIds()) {
      Formal.dump();
      std::cout << " ";
    }
    std::cout << ")";
    break;
  }
  case Formal::Type::ListRest: {
    ListRestFormal const &FListRest = static_cast<ListRestFormal const &>(F);
    std::cout << "(";
    for (const auto &Formal : FListRest.getIds()) {
      Formal.dump();
      std::cout << " ";
    }
    std::cout << " . ";
    FListRest.getRestFormal().dump();
    std::cout << ")";
    break;
  }
  }
  std::wcout << L" ";
  Body->dump();
  std::wcout << L")";
}

void Lambda::write() const { std::wcout << L"#<procedure>"; }

//
// Implementation of DefineValues node.
//
DefineValues::DefineValues(const DefineValues &DV)
    : ClonableNode(ASTNodeKind::AST_DefineValues), Body(DV.Body->clone()) {
  Ids.reserve(DV.Ids.size());
  for (const auto &Id : DV.Ids) {
    Ids.emplace_back(Id);
  }
}

DefineValues::DefineValues(std::vector<Identifier> Ids,
                           std::unique_ptr<ExprNode> &Body)
    : ClonableNode(ASTNodeKind::AST_DefineValues), Ids(std::move(Ids)),
      Body(std::move(Body)) {}

const ExprNode &DefineValues::getBody() const { return *Body; }

void DefineValues::dump() const {
  std::wcout << L"(define-values (";
  for (const auto &Id : Ids) {
    std::wcout << Id.getName() << L" ";
  }
  std::wcout << L") ";
  Body->dump();
  std::wcout << L")";
}

//
// Implementation of Linklet node.
//
Linklet::Linklet(const Linklet &L) : ClonableNode(ASTNodeKind::AST_Linklet) {
  // Need to perform a deep copy of imports, exports and the whole body.

  // Copy each of the import and export vectors
  for (const auto &I : L.Imports) {
    Imports.emplace_back(std::make_pair(I.first, I.second));
  }
  for (const auto &E : L.Exports) {
    Exports.emplace_back(std::make_pair(E.first, E.second));
  }

  // Deep copy the body
  Body.reserve(L.Body.size());
  for (const auto &Form : L.Body) {
    Body.emplace_back(Form->clone());
  }
}

Linklet::FormRange::FormRange(
    std::vector<std::unique_ptr<ast::TLNode>>::const_iterator FsBegin,
    std::vector<std::unique_ptr<ast::TLNode>>::const_iterator FsEnd)
    : BeginIt(FsBegin), EndIt(FsEnd) {}

TLNode const &Linklet::FormRange::operator[](size_t I) const {
  return *BeginIt[I];
}

void Linklet::appendImport(const Identifier &ExtId, const Identifier &IntId) {
  Imports.emplace_back(std::make_pair(ExtId, IntId));
}
void Linklet::appendExport(const Identifier &IntId, const Identifier &ExtId) {
  Exports.emplace_back(std::make_pair(IntId, ExtId));
}
void Linklet::appendBodyForm(std::unique_ptr<TLNode> &&Form) {
  Body.emplace_back(std::move(Form));
}

void Linklet::dump() const {
  std::wcout << L"(linklet (";
  for (const auto &I : Imports) {
    std::wcout << L"(" << I.first.getName() << L" " << I.second.getName()
               << L") ";
  }
  std::wcout << L") (";
  for (const auto &E : Exports) {
    std::wcout << L"(" << E.first.getName() << L" " << E.second.getName()
               << L") ";
  }
  std::wcout << L") (";
  for (const auto &Form : Body) {
    Form->dump();
    std::wcout << L" ";
  }
  std::wcout << L"))";
}

//
// Implementation of SetBang node.
//

SetBang::SetBang(const SetBang &SB)
    : ClonableNode(ASTNodeKind::AST_SetBang),
      Expr(std::unique_ptr<ast::ExprNode>(SB.Expr->clone())) {
  Id = std::unique_ptr<ast::Identifier>(new ast::Identifier(*SB.Id));
}

Identifier const &SetBang::getIdentifier() const { return *Id; }
ExprNode const &SetBang::getExpr() const { return *Expr; }

void SetBang::setIdentifier(std::unique_ptr<Identifier> &&I) {
  Id = std::move(I);
}
void SetBang::setExpr(std::unique_ptr<ExprNode> &&E) { Expr = std::move(E); }

//
// Implementation of Values node.
//

Values::Values(std::vector<std::unique_ptr<ExprNode>> Exprs)
    : ClonableNode(ASTNodeKind::AST_Values), Exprs(std::move(Exprs)) {}

// Copy constructor for values.
Values::Values(const Values &V) : ClonableNode(ASTNodeKind::AST_Values) {
  for (const auto &Expr : V.getExprs()) {
    Exprs.emplace_back(Expr->clone());
  }
}

Values::ExprRange::ExprRange(
    std::vector<std::unique_ptr<ast::ExprNode>>::const_iterator EsBegin,
    std::vector<std::unique_ptr<ast::ExprNode>>::const_iterator EsEnd)
    : BeginIt(EsBegin), EndIt(EsEnd) {}

ExprNode const &Values::ExprRange::operator[](size_t I) const {
  return *BeginIt[I];
}

void Values::dump() const {
  std::wcout << L"(values";
  for (const auto &Expr : Exprs) {
    std::wcout << L" ";
    Expr->dump();
  }
  std::wcout << L")";
}

void Values::write() const {
  for (const auto &Expr : Exprs) {
    ExprNode *E = Expr.get();
    if (ValueNode *V = llvm::dyn_cast<ValueNode>(E)) {
      V->write();
      std::wcout << std::endl;
    } else {
      std::wcerr << L"Error: non-value in values expression" << std::endl;
      exit(1);
    }
  }
}

//
// Implementation of LetValues node.
//

LetValues::LetValues(const LetValues &DV)
    : ClonableNode(ASTNodeKind::AST_LetValues) {
  for (auto const &Id : DV.Ids) {
    Ids.emplace_back(Id);
  }
  for (auto const &Expr : DV.Exprs) {
    Exprs.emplace_back(std::unique_ptr<ast::ExprNode>(Expr->clone()));
  }
  for (auto const &Expr : DV.Body) {
    Body.emplace_back(std::unique_ptr<ast::ExprNode>(Expr->clone()));
  }
}

void LetValues::appendBinding(std::vector<Identifier> &&Ids,
                              std::unique_ptr<ExprNode> Expr) {
  this->Ids.emplace_back(std::move(Ids));
  this->Exprs.emplace_back(std::move(Expr));
}

void LetValues::appendBody(std::unique_ptr<ExprNode> Expr) {
  Body.emplace_back(std::move(Expr));
}
ast::LetValues::IdRange LetValues::getBindingIds(size_t Idx) const {
  assert(Idx < Ids.size());
  return IdRange{Ids[Idx]};
}
ExprNode const &LetValues::getBindingExpr(size_t Idx) const {
  return *Exprs[Idx];
}
ExprNode const &LetValues::getBodyExpr(size_t Idx) const { return *Body[Idx]; }
size_t LetValues::exprsCount() const { return Exprs.size(); }

//
// Implementation of Void node.
//
void Void::dump() const { std::wcerr << "(void)"; }

void Void::write() const {
  // Do nothing
}

//
// Implementation of List node.
//
List::List(List const &L) : ClonableNode(ASTNodeKind::AST_List) {
  for (auto &V : L.Values) {
    Values.emplace_back(std::unique_ptr<ValueNode>(V->clone()));
  }
}

void List::appendExpr(std::unique_ptr<ValueNode> &&Value) {
  Values.emplace_back(std::move(Value));
}

ValueNode const &List::operator[](size_t I) const { return *Values[I]; }

void List::dump() const {
  const List &L = *this;
  std::cout << "(list ";
  for (size_t i = 0; i < L.length(); ++i) {
    L[i].dump();
    if (i != L.length() - 1)
      std::cout << " ";
  }
  std::cout << ")";
}

void List::write() const {
  const List &L = *this;
  std::cout << "(";
  for (size_t I = 0; I < L.length(); ++I) {
    L[I].write();
    if (I != L.length() - 1)
      std::cout << " ";
  }
  std::cout << ")";
}
