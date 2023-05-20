#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ConvertUTF.h>

namespace UTF8 {

// Returns true is the character c has a graphical representation in UTF8.
bool isGraphUTF8(const llvm::UTF8 *Start, const llvm::UTF8 *End);

// Returns true if StringRef Str starts with a UTF8 replacement character.
bool isUTF8ReplacementCharacter(const llvm::UTF8 *Start, const llvm::UTF8 *End);

}; // namespace UTF8