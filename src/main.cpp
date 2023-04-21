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

#include "config.h"
#include "interpreter.h"
#include "parse.h"

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
  std::unique_ptr<ast::Linklet> AST = parseLinklet(Input);

  if (!AST) {
    std::cerr << "Parsing failed!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (Verbose) {
    std::cout << "Parsing successful!" << std::endl;
    std::cout << "Dumping AST:" << std::endl;
    AST->dump();
  }

  Interpreter I;
  AST->accept(I);
  auto const &Result = I.getResult();

  if (!Result) {
    std::cerr << "Interpretation failed!" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (Verbose)
    std::cout << "Interpretation successful!" << std::endl;

  Result->write();
  std::cout << std::endl;

  return 0;
}