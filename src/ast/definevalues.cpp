#include "ast/definevalues.h"

#include <utility>

#include "ast/arithplus.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/values.h"
#include "ast/void.h"

using namespace nir;

DefineValues::DefineValues(std::vector<Identifier> Ids,
                           std::unique_ptr<ExprNode> &Body)
    : Ids(std::move(Ids)), Body(std::move(Body)) {}

const ExprNode &DefineValues::getBody() const { return *Body; }