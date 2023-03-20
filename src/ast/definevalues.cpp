#include "ast/definevalues.h"

#include <utility>

#include "toplevelnode_inc.h"

using namespace nir;

DefineValues::DefineValues(std::vector<Identifier> Ids,
                           std::unique_ptr<ExprNode> &Body)
    : Ids(std::move(Ids)), Body(std::move(Body)) {}

const ExprNode &DefineValues::getBody() const { return *Body; }