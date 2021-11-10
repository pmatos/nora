#pragma once

#include <memory>
#include <vector>

#include "identifier.h"
#include "toplevelnode.h"

namespace nir {

class Linklet {
public:
  // helper type
  using idpair_t = std::pair<Identifier, Identifier>;

  Linklet() = default;
  Linklet(const Linklet &L) = delete;
  Linklet(Linklet &&L) = default;
  Linklet &operator=(const Linklet &L) = delete;
  Linklet &operator=(Linklet &&L) = default;

  ~Linklet() = default;

  class FormRange {
  public:
    FormRange() = delete;
    FormRange(std::vector<std::unique_ptr<TLNode>>::const_iterator FsBegin,
              std::vector<std::unique_ptr<TLNode>>::const_iterator FsEnd);
    [[nodiscard]] auto begin() const { return BeginIt; }
    [[nodiscard]] auto end() const { return EndIt; }
    [[nodiscard]] TLNode const &operator[](size_t I) const;

  private:
    std::vector<std::unique_ptr<TLNode>>::const_iterator BeginIt, EndIt;
  };

  void appendImport(const Identifier &ExtId, const Identifier &IntId);
  void appendExport(const Identifier &IntId, const Identifier &ExtId);
  void appendBodyForm(std::unique_ptr<TLNode> &&Form);

  size_t exportsCount() const { return Exports.size(); }
  size_t importsCount() const { return Imports.size(); }
  size_t bodyFormsCount() const { return Body.size(); }

  [[nodiscard]] FormRange getBody() const {
    return {Body.cbegin(), Body.cend()};
  };
  [[nodiscard]] const std::vector<idpair_t> &getImports() const {
    return Imports;
  }
  [[nodiscard]] const std::vector<idpair_t> &getExports() const {
    return Exports;
  }

private:
  // Sets of imports. The pair contains two Ids
  // One for external-imported-id and the other for internal-imported-id
  std::vector<idpair_t> Imports;

  // Sets of exports. The pair contains two Ids
  // One for internal-exported-id and the other for external-exported-id
  // If there is just a single exported-id, then both are the same
  std::vector<idpair_t> Exports;

  /// Linklet body expressions
  std::vector<std::unique_ptr<TLNode>> Body;
};

}; // namespace nir