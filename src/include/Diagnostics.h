#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>

namespace nora {

// A thin wrapper over llvm::SourceMgr that renders NORA diagnostics (with the
// familiar file:line:col, source line and caret) and keeps track of how many
// errors have been reported. Source buffers are registered with addBuffer so
// that SMLocs pointing into them render with location information; locations
// are produced from byte offsets by SourceStream::getLoc.
class DiagnosticEngine {
public:
  DiagnosticEngine() = default;

  // Register a buffer's contents (a non-owning view) with the underlying
  // SourceMgr so that locations within it can be rendered. Returns the
  // SourceMgr buffer id.
  unsigned addBuffer(llvm::StringRef Contents, llvm::StringRef Name);

  // Report a diagnostic anchored at a source location. An invalid Loc prints
  // the message without location information.
  void error(llvm::SMLoc Loc, const llvm::Twine &Msg,
             llvm::ArrayRef<llvm::SMRange> Ranges = {});
  void warning(llvm::SMLoc Loc, const llvm::Twine &Msg,
               llvm::ArrayRef<llvm::SMRange> Ranges = {});
  void note(llvm::SMLoc Loc, const llvm::Twine &Msg,
            llvm::ArrayRef<llvm::SMRange> Ranges = {});

  // Report a diagnostic with no associated source location (e.g. I/O errors).
  void error(const llvm::Twine &Msg);

  [[nodiscard]] bool hadError() const { return NumErrors != 0; }
  [[nodiscard]] unsigned getNumErrors() const { return NumErrors; }
  [[nodiscard]] unsigned getNumWarnings() const { return NumWarnings; }

  llvm::SourceMgr &getSourceMgr() { return SrcMgr; }

private:
  llvm::SourceMgr SrcMgr;
  unsigned NumErrors = 0;
  unsigned NumWarnings = 0;
};

} // namespace nora
