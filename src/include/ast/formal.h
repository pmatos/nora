#pragma once

#include <memory>
#include <vector>

#include "ast/identifier.h"

namespace nir {

// There are three types of lambda formals:
// 1. A formal that is an identifier. This is when we want to pass a single
//   list into the lambda.
//   (lambda x (do-something x)) , and I can pass as many values as I want and
//   they are all aggregated into a list.
// 2. A formal that is a list of identifiers. This is when we want to pass a
// specific
//    number of arguments into the lambda and each argumend is bound to a
//    formal.
// 3. A formal that is a list of identifiers and a rest identifier. This is when
// we want to
//    to pass a minimum number of arguments into the lambda. Each argument up to
//    the number of formals is bound to a formal. The rest of the arguments are
//    bound to the rest identifier as a list.
class Formal {
public:
  enum class Type { Identifier, List, ListRest };
  virtual ~Formal() = default;
  virtual std::unique_ptr<Formal> clone() const = 0;

  virtual Type getType() const = 0;
};

class IdentifierFormal : public Formal {
public:
  IdentifierFormal(const Identifier &I) : I(I){};
  IdentifierFormal(const IdentifierFormal &other) = default;
  IdentifierFormal(IdentifierFormal &&other) = default;
  IdentifierFormal &operator=(const IdentifierFormal &other) = delete;
  IdentifierFormal &operator=(IdentifierFormal &&other) = default;
  ~IdentifierFormal() = default;

  std::unique_ptr<Formal> clone() const override {
    return std::make_unique<IdentifierFormal>(*this);
  }

  [[nodiscard]] Type getType() const override { return Type::Identifier; }
  [[nodiscard]] const Identifier &getIdentifier() const { return I; }

private:
  Identifier I;
};

class ListFormal : public Formal {
public:
  ListFormal() = default;
  ListFormal(const std::vector<Identifier> &Ids) : Formals(Ids) {}
  ListFormal(ListFormal const &other) = default;
  ListFormal(ListFormal &&other) = default;
  ListFormal &operator=(const ListFormal &other) = delete;
  ListFormal &operator=(ListFormal &&other) = default;
  ~ListFormal() override = default;

  std::unique_ptr<Formal> clone() const override {
    return std::make_unique<ListFormal>(*this);
  }

  class IdRange {
  public:
    IdRange() = delete;
    IdRange(const std::vector<Identifier> &Ids)
        : BeginIt(Ids.cbegin()), EndIt(Ids.cend()) {}
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }

  private:
    std::vector<Identifier>::const_iterator BeginIt, EndIt;
  };

  IdRange getIds() const { return IdRange{Formals}; }
  [[nodiscard]] Type getType() const override { return Type::List; }
  void addFormal(const Identifier &I);

private:
  std::vector<Identifier> Formals;
};

class ListRestFormal : public ListFormal {
public:
  ListRestFormal(const std::vector<Identifier> &Formals,
                 const Identifier &RestFormal)
      : ListFormal(Formals), RestFormal(RestFormal) {}
  ListRestFormal(const ListRestFormal &other) = default;
  ListRestFormal(ListRestFormal &&other) = default;
  ListRestFormal &operator=(const ListRestFormal &other) = delete;
  ListRestFormal &operator=(ListRestFormal &&other) = default;
  ~ListRestFormal() override = default;

  std::unique_ptr<Formal> clone() const override {
    return std::make_unique<ListRestFormal>(*this);
  }
  [[nodiscard]] Type getType() const override { return Type::ListRest; }
  void addRestFormal(const Identifier &I);
  const Identifier &getRestFormal() const { return RestFormal; }

private:
  Identifier RestFormal;
};
}; // namespace nir