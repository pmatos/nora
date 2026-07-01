#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SMLoc.h>

#include <regex>
#include <string>

namespace nora {
class DiagnosticEngine;
} // namespace nora

class SourceStream {
public:
  static llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getFileBuffer(const std::string &Path);

  explicit SourceStream(const std::string &Path,
                        nora::DiagnosticEngine *Diag = nullptr);
  explicit SourceStream(const char *View,
                        nora::DiagnosticEngine *Diag = nullptr);

  [[nodiscard]] char peekChar(size_t N = 0) const;
  void skipPrefix(size_t N) { Position += N; }
  [[nodiscard]] llvm::StringRef getSubview(size_t N) const {
    return Contents.substr(Position, N);
  }
  llvm::StringRef getSubviewAndSkip(size_t N) {
    auto View = getSubview(N);
    skipPrefix(N);
    return View;
  }

  void rewind(size_t Len);
  void rewindTo(size_t Pos);

  void skipWhitespace();
  bool searchRegex(const char *Regex, std::cmatch &CM) const;
  bool isPrefix(const char *Str) const;
  [[nodiscard]] size_t getPosition() const;
  [[nodiscard]] bool isEmpty() const { return Position >= Contents.size(); }

  // Source location for a byte offset into the stream, for diagnostics. The
  // returned SMLoc points into the same bytes registered with the
  // DiagnosticEngine, so it renders with line/column information.
  [[nodiscard]] llvm::SMLoc getLoc(size_t Offset) const {
    return llvm::SMLoc::getFromPointer(Contents.data() + Offset);
  }
  [[nodiscard]] nora::DiagnosticEngine *getDiagnostics() const { return Diag; }

private:
  std::string Path;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  llvm::StringRef Contents; // Contents of Buffer as a read-only string ref
  size_t Position;          // Current parse position in Contents
  nora::DiagnosticEngine *Diag = nullptr; // Diagnostics sink (may be null)
};