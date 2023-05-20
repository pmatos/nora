#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <regex>
#include <string>

class SourceStream {
public:
  static llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getFileBuffer(const std::string &Path);

  explicit SourceStream(const std::string &Path);
  explicit SourceStream(const char *View);

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

private:
  std::string Path;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;
  llvm::StringRef Contents; // Contents of Buffer as a read-only string ref
  size_t Position;          // Current parse position in Contents
};