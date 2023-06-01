#pragma once

#include "SourceStream.h"
#include "ast.h"

#include <llvm/Support/Errc.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

namespace Parse {

bool isExplicitSign(SourceStream &S, size_t Offset = 0);
bool isSubsequent(SourceStream &S, size_t Offset = 0);
bool isDigit(SourceStream &S, size_t Offset = 0);
bool isHexDigit(SourceStream &S, size_t Offset = 0);
size_t isHexScalarValue(SourceStream &S, size_t Offset = 0);
bool isLetter(SourceStream &S, size_t Offset = 0);
bool isSpecialInitial(SourceStream &S, size_t Offset = 0);

// Parsing functions:
std::unique_ptr<ast::ExprNode> parseExpr(SourceStream &S);
std::unique_ptr<ast::Lambda> parseLambda(SourceStream &S);
std::unique_ptr<ast::Linklet> parseLinklet(SourceStream &S);
std::unique_ptr<ast::Begin> parseBegin(SourceStream &S);
std::unique_ptr<ast::TLNode> parseDefn(SourceStream &S);
std::unique_ptr<ast::ExprNode> parseExpr(SourceStream &S);
std::unique_ptr<ast::Identifier> parseIdentifier(SourceStream &S);
std::unique_ptr<ast::Values> parseValues(SourceStream &S);
std::unique_ptr<ast::DefineValues> parseDefineValues(SourceStream &S);
std::unique_ptr<ast::Formal> parseFormals(SourceStream &S);
std::unique_ptr<ast::Application> parseApplication(SourceStream &S);
std::unique_ptr<ast::SetBang> parseSetBang(SourceStream &S);
std::unique_ptr<ast::IfCond> parseIfCond(SourceStream &S);
std::unique_ptr<ast::BooleanLiteral> parseBooleanLiteral(SourceStream &S);
std::unique_ptr<ast::LetValues> parseLetValues(SourceStream &S);
std::unique_ptr<ast::TLNode> parseDefnOrExpr(SourceStream &S);
std::unique_ptr<ast::Integer> parseInteger(SourceStream &S);

}; // namespace Parse