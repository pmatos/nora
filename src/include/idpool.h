#pragma once

#include <set>
#include <string>
#include <string_view>

#include "ast.h"

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

  ast::Identifier create(const std::string &Name);
  ast::Identifier create(const ast::Identifier &Id);

private:
  static IdPool *Inst;
  std::set<std::string> StrPool;
};