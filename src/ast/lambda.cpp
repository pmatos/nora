#include "ast/lambda.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "ast/formal.h"
#include "exprnode_inc.h"

using namespace nir;

// Copy Constructor for Lambda.
Lambda::Lambda(Lambda const &L) {
  Formals = L.Formals->clone();
  Body = std::make_unique<ExprNode>(*(L.Body));
}

const ExprNode &Lambda::getBody() const { return *Body; }
