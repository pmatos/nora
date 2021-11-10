#include "utils/idpool.h"

IdPool *IdPool::Inst = nullptr;

nir::Identifier IdPool::create(const std::wstring &Name) {
  std::pair<std::set<std::wstring>::iterator, bool> V = StrPool.insert(Name);
  return nir::Identifier{std::wstring_view(*(V.first))};
}