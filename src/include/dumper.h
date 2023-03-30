#pragma once

#include <cassert>

#include "ast/formal.h"
#include "astnode.h"

struct Dumper {
  void operator()(nir::Identifier const &Id);
  void operator()(nir::Integer const &Int);
  void operator()(nir::Linklet const &Linklet);
  void operator()(nir::DefineValues const &DV);
  void operator()(nir::Values const &DV);
  void operator()(nir::ArithPlus const &DV);
  void operator()(nir::Void const &Vd);
  void operator()(nir::Lambda const &L);
  void operator()(nir::Formal const &F);
  void operator()(nir::Begin const &B);
  void operator()(nir::List const &L);
  void operator()(nir::Application const &A);
  void operator()(nir::SetBang const &SB);
};
