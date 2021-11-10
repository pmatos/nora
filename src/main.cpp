#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <variant>

#include "ast/arithplus.h"
#include "ast/definevalues.h"
#include "ast/identifier.h"
#include "ast/integer.h"
#include "ast/lambda.h"
#include "ast/values.h"
#include "ast/void.h"
#include "config.h"
#include "dumper.h"
#include "interpreter.h"
#include "parse.h"
#include "valuenode.h"

namespace fs = std::filesystem;

static bool Verbose = false;

int main(int argc, char *argv[]) {
  if (Verbose)
    std::cout << "Nora pre-release " << PROJECT_VERSION << std::endl;

  if (argc != 2) {
    exit(EXIT_FAILURE);
  }

  // Initialize logger
  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(plog::none, &consoleAppender);

  fs::path Path = argv[1];
  if (Verbose)
    std::cout << "Parsing linklet in file " << Path << std::endl;

  Stream Input(Path);
  std::unique_ptr<nir::Linklet> AST = parseLinklet(Input);

  if (!AST) {
    std::cerr << "Parsing failed!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (Verbose) {
    std::cout << "Parsing successful!" << std::endl;
    std::cout << "Dumping AST:" << std::endl;
    Dumper Dump;
    Dump(*AST);
  }

  Interpreter I;
  std::unique_ptr<nir::ValueNode> Result = I(*AST);

  if (!Result) {
    std::cerr << "Interpretation failed!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (Verbose)
    std::cout << "Interpretation successful!" << std::endl;

  std::visit(Dumper{}, *Result);
  std::cout << std::endl;

  return 0;
}