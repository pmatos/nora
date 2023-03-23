#include "ast/application.h"

#include "exprnode_inc.h"

using namespace nir;

// Copy Constructor for Application.
Application::Application(const Application &A) {
  for (const auto &Expr : A.Exprs) {
    std::unique_ptr<ExprNode> Ptr = std::make_unique<ExprNode>(*Expr);
    Exprs.emplace_back(std::move(Ptr));
  }
}

void Application::appendExpr(std::unique_ptr<ExprNode> &&E) {
  Exprs.emplace_back(std::move(E));
}

ExprNode const &Application::operator[](size_t I) const { return *Exprs[I]; }