#pragma once

#include <memory>
#include <vector>

#include "exprnode.h"
#include "formal.h"

namespace nir {
class Lambda {
public:
  Lambda() = default;
  Lambda(Lambda const &L);
  Lambda(Lambda &&L) = default;
  Lambda &operator=(const Lambda &L) = delete;
  Lambda &operator=(Lambda &&L) = default;
  ~Lambda() = default;

  void setFormals(std::unique_ptr<Formal> F) { Formals = std::move(F); }
  void setBody(std::unique_ptr<ExprNode> E) { Body = std::move(E); }

  Formal::Type getFormalsType() const { return Formals->getType(); }

  [[nodiscard]] Formal const &getFormals() const { return *Formals; }
  [[nodiscard]] const ExprNode &getBody() const;

private:
  std::unique_ptr<Formal> Formals;
  std::unique_ptr<ExprNode> Body;
};
}; // namespace nir