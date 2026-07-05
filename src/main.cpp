#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <variant>

#include <gc.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/InitLLVM.h>

#include "Diagnostics.h"
#include "Interpreter.h"
#include "Parse.h"
#include "SourceStream.h"
#include "config.h"

namespace cl = llvm::cl;

namespace {
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::opt<bool> Verbose("v", cl::desc("Enable verbose output"),
                             cl::init(false));
static cl::opt<bool, true> DebugFlag("debug", cl::desc("Enable debug output"),
                                     cl::Hidden, cl::location(llvm::DebugFlag));

enum Action { None, DumpAST };

static cl::opt<enum Action>
    EmitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

} // namespace

int main(int argc, char *argv[]) {
  // Bring up the Boehm collector before anything allocates so it records the
  // main-thread stack bottom (M2 value model). Non-incremental (the default),
  // and before llvm::InitLLVM installs its signal handlers.
  GC_INIT();
  llvm::InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "norac\n");

  if (Verbose) {
    std::cout << "Nora pre-release " << PROJECT_VERSION << std::endl;
    std::cout << "Parsing linklet in file " << InputFilename << std::endl;
  }

  nora::DiagnosticEngine Diags;
  SourceStream Input(InputFilename, &Diags);
  std::unique_ptr<ast::Linklet> AST = Parse::parseLinklet(Input);

  if (!AST || Diags.hadError()) {
    // parseLinklet reports a precise diagnostic on failure; guard against the
    // unlikely case where it returned null without emitting one.
    if (!Diags.hadError()) {
      Diags.error("failed to parse linklet");
    }
    return EXIT_FAILURE;
  }

  if (Verbose) {
    std::cout << "Parsing successful!" << std::endl;
  }

  if (EmitAction == Action::DumpAST) {
    std::cout << "Dumping AST:" << std::endl;
    AST->dump();
  }

  Interpreter I(Diags);
  AST->accept(I);

  if (Diags.hadError()) {
    return EXIT_FAILURE;
  }
  if (Verbose)
    std::cout << "Interpretation successful!" << std::endl;

  I.getResult()->write();
  std::cout << std::endl;

  return 0;
}