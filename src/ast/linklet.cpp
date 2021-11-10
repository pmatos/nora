#include "ast/linklet.h"

#include <memory>

#include "ast/arithplus.h"
#include "ast/definevalues.h"
#include "ast/identifier.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/values.h"
#include "ast/void.h"

using namespace nir;

Linklet::FormRange::FormRange(
    std::vector<std::unique_ptr<nir::TLNode>>::const_iterator FsBegin,
    std::vector<std::unique_ptr<nir::TLNode>>::const_iterator FsEnd)
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