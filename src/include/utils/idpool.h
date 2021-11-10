#pragma once

#include <set>
#include <string>
#include <string_view>

#include "ast/identifier.h"

// Identifier nodes are kept in a Pool and handles
// are passed around
// Pool is a singleton class that can be access through the method
// instance();
class IdPool {
public:
  static IdPool &instance() {
    if (!Inst) {
      Inst = new IdPool();
    }
    return *Inst;
  }

  nir::Identifier create(const std::wstring &Name);

private:
  static IdPool *Inst;
  std::set<std::wstring> StrPool;
};