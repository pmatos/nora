#include "SourceStream.h"

SourceStream::SourceStream(const std::string &Path) : Path(Path) {
  auto MaybeBuffer = SourceStream::getFileBuffer(Path);
  if (!MaybeBuffer) {
    llvm::errs() << "Error reading file: " << Path << "\n";
    exit(EXIT_FAILURE);
  }
  Buffer = std::move(*MaybeBuffer);
  Contents = Buffer->getBuffer();
  Position = 0;
}

SourceStream::SourceStream(const char *View)
    : Path(), Contents(View), Position(0) {}

/// Read the file pointed to by \param path and returns its contents as a
/// MemoryBuffer.
llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
SourceStream::getFileBuffer(const std::string &Filename) {
  if (Filename == "-") {
    return llvm::MemoryBuffer::getSTDIN();
  } else {
    return llvm::MemoryBuffer::getFile(Filename);
  }
}

size_t SourceStream::getPosition() const { return Position; }

bool SourceStream::searchRegex(const char *Regex, std::cmatch &CM) const {
  std::regex E(Regex);

  auto Window = Contents.drop_front(Position);
  return std::regex_search(Window.begin(), Window.end(), CM, E,
                           std::regex_constants::match_continuous);
}

bool SourceStream::isPrefix(const char *Str) const {
  return Contents.drop_front(Position).startswith(Str);
}

char SourceStream::peekChar(size_t N) const {
  if (Position + N < Contents.size()) {
    return Contents[Position + N];
  }
  return EOF;
}

/// Skips whitespace in SourceStream
void SourceStream::skipWhitespace() {
  if (isEmpty()) {
    return;
  }

  // Lets eat comments now
  if (peekChar() == ';') {
    while (peekChar() != '\n' && !isEmpty()) {
      skipPrefix(1);
    }
  }

  // Lets eat spaces - including the previously left \n at the end of the line
  // comment
  while (isspace(Contents[Position])) {
    skipPrefix(1);
  }

  if (Contents[Position] == ';') {
    skipWhitespace();
  }
}

void SourceStream::rewind(size_t Len) { Position -= Len; }
void SourceStream::rewindTo(size_t Pos) { Position = Pos; }
