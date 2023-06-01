#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <llvm/Support/CommandLine.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <variant>

#include "Parse.h"
#include "config.h"
#include "interpreter.h"

namespace cl = llvm::cl;

namespace {
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::opt<bool> Verbose("v", cl::desc("Enable verbose output"),
                             cl::init(false));

enum Action { None, DumpAST };

static cl::opt<enum Action>
    EmitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

} // namespace

int main(int argc, char *argv[]) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "norac\n");

  if (Verbose) {
    std::cout << "Nora pre-release " << PROJECT_VERSION << std::endl;
    std::cout << "Parsing linklet in file " << InputFilename << std::endl;
  }

  SourceStream Input(InputFilename);
  std::unique_ptr<ast::Linklet> AST = Parse::parseLinklet(Input);

  if (!AST) {
    std::cerr << "Parsing failed!" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (Verbose) {
    std::cout << "Parsing successful!" << std::endl;
  }

  if (EmitAction == Action::DumpAST) {
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