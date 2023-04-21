#include "idpool.h"

#include "ast.h"

IdPool *IdPool::Inst = nullptr;

ast::Identifier IdPool::create(const std::wstring &Name) {
  std::pair<std::set<std::wstring>::iterator, bool> V = StrPool.insert(Name);
  return ast::Identifier{std::wstring_view(*(V.first))};
}

ast::Identifier IdPool::create(const ast::Identifier &Id) {
  const std::wstring Name(Id.getName());
  return create(Name);
}