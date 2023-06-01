#include "idpool.h"

#include "ast.h"

IdPool *IdPool::Inst = nullptr;

ast::Identifier IdPool::create(const std::string &Name) {
  std::pair<std::set<std::string>::iterator, bool> V = StrPool.insert(Name);
  return ast::Identifier{std::string_view(*(V.first))};
}

ast::Identifier IdPool::create(const ast::Identifier &Id) {
  const std::string Name(Id.getName());
  return create(Name);
}