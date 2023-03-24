#include "dumper.h"

#include <gmp.h>
#include <iostream>

#include "ast/formal.h"
#include "ast/linklet.h"
#include "toplevelnode_inc.h"

void Dumper::operator()(nir::Identifier const &Id) {
  std::wcout << Id.getName();
}

void Dumper::operator()(nir::Integer const &Int) { Int.dump(); }

void Dumper::operator()(nir::Values const &V) {
  for (const auto &Arg : V.getExprs()) {
    std::visit(*this, *Arg);
    std::cout << std::endl;
  }
}

void Dumper::operator()(nir::DefineValues const &DV) {}

void Dumper::operator()(nir::ArithPlus const &AP) {}

void Dumper::operator()(nir::Void const &Vd) {
  // do nothing
}

void Dumper::operator()(nir::Lambda const &L) {
  Dumper Dump;

  std::cout << "(lambda ( ";

  Dump(L.getFormals());

  std::cout << ") ";
  std::visit(Dump, L.getBody());
  std::cout << ")";
}

void Dumper::operator()(nir::Formal const &F) {
  Dumper Dump;

  switch (F.getType()) {
  case nir::Formal::Type::Identifier: {
    nir::IdentifierFormal const &FId =
        static_cast<nir::IdentifierFormal const &>(F);
    Dump(FId.getIdentifier());
    break;
  }
  case nir::Formal::Type::List: {
    nir::ListFormal const &FList = static_cast<nir::ListFormal const &>(F);
    std::cout << "(";
    for (const auto &Formal : FList.getIds()) {
      Dump(Formal);
      std::cout << " ";
    }
    std::cout << ")";
    break;
  }
  case nir::Formal::Type::ListRest: {
    nir::ListRestFormal const &FListRest =
        static_cast<nir::ListRestFormal const &>(F);
    std::cout << "(";
    for (const auto &Formal : FListRest.getIds()) {
      Dump(Formal);
      std::cout << " ";
    }
    std::cout << " . ";
    Dump(FListRest.getRestFormal());
    std::cout << ")";
    break;
  }
  }
}

void Dumper::operator()(nir::Linklet const &Linklet) {
  Dumper Dump;
  std::cout << "(linklet ()";

  std::cout << " (";

  // Dump exports
  for (const auto &Idpair : Linklet.getExports()) {
    nir::Identifier IntName = Idpair.first;
    nir::Identifier ExtName = Idpair.second;

    std::cout << "(";
    Dump(IntName);
    std::cout << " ";
    Dump(ExtName);
    std::cout << ")";
  }
  std::cout << ") ";

  for (const auto &BodyForm : Linklet.getBody()) {
    std::visit(Dump, *BodyForm);
  }

  std::cout << ")" << std::endl;
}

void Dumper::operator()(nir::Begin const &B) {
  Dumper Dump;
  std::cout << "(";

  if (B.isZero())
    std::cout << "begin0 ";
  else
    std::cout << "begin ";

  for (const auto &Expr : B.getBody()) {
    std::visit(Dump, *Expr);
    std::cout << " ";
  }
  std::cout << ")";
}

void Dumper::operator()(nir::List const &L) {
  Dumper Dump;
  std::cout << "(";
  for (size_t i = 0; i < L.length(); ++i) {
    std::visit(Dump, L[i]);
    std::cout << " ";
  }
  std::cout << ")";
}