#include "UTF8.h"

bool UTF8::isGraphUTF8(const llvm::UTF8 *Start, const llvm::UTF8 *End) {
  if (Start == End) { // it's ascii
    return std::isgraph(*Start);
  }

  llvm::UTF8 codePoint = *Start;
  unsigned Bytes = llvm::getNumBytesForUTF8(codePoint);

  // If the number of bytes is more than the distance between
  // Start and End, then the character is not complete.
  if (Bytes > (unsigned)(End - Start + 1)) {
    return false;
  }

  // Check if the character is ASCII.
  if (Bytes == 1) {
    return (codePoint >= 0x20 && codePoint <= 0x7E);
  }

  // Check the first byte of the character.
  if (codePoint < 0xC2 || codePoint > 0xF4) {
    return false;
  }

  // Check the remaining bytes of the character.
  for (unsigned i = 1; i < Bytes; ++i) {
    codePoint = *(Start + i);
    if (codePoint < 0x80 || codePoint > 0xBF) {
      return false;
    }
  }

  return true;
}

bool UTF8::isUTF8ReplacementCharacter(const llvm::UTF8 *Start,
                                      const llvm::UTF8 *End) {
  // The unicode replacement character is 0xFFFD.
  // Lets check if the Str starts with the UTF8 representation of 0xFFFD.
  if (End - Start != 1) {
    return false;
  }

  return *Start == u'\xFF' && *End == u'\xFD';
}