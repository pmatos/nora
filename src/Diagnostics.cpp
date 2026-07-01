#include "Diagnostics.h"

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/WithColor.h>
#include <llvm/Support/raw_ostream.h>

namespace nora {

unsigned DiagnosticEngine::addBuffer(llvm::StringRef Contents,
                                     llvm::StringRef Name) {
  // A non-owning MemoryBuffer view: SourceStream keeps ownership of the real
  // buffer, so the SMLocs produced from its Contents point into the same bytes
  // the SourceMgr renders.
  return SrcMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(Contents, Name,
                                       /*RequiresNullTerminator=*/false),
      llvm::SMLoc());
}

void DiagnosticEngine::error(llvm::SMLoc Loc, const llvm::Twine &Msg,
                             llvm::ArrayRef<llvm::SMRange> Ranges) {
  ++NumErrors;
  SrcMgr.PrintMessage(Loc, llvm::SourceMgr::DK_Error, Msg, Ranges);
}

void DiagnosticEngine::warning(llvm::SMLoc Loc, const llvm::Twine &Msg,
                               llvm::ArrayRef<llvm::SMRange> Ranges) {
  ++NumWarnings;
  SrcMgr.PrintMessage(Loc, llvm::SourceMgr::DK_Warning, Msg, Ranges);
}

void DiagnosticEngine::note(llvm::SMLoc Loc, const llvm::Twine &Msg,
                            llvm::ArrayRef<llvm::SMRange> Ranges) {
  SrcMgr.PrintMessage(Loc, llvm::SourceMgr::DK_Note, Msg, Ranges);
}

void DiagnosticEngine::error(const llvm::Twine &Msg) {
  ++NumErrors;
  llvm::WithColor::error(llvm::errs(), "norac") << Msg << '\n';
}

} // namespace nora
