// Shared Catch2 entry point for the unit test executables. GC_INIT() must run
// on the main thread before any test allocates, so the Boehm collector records
// the correct stack bottom. (The GMP allocator hook lands in a later GC slice.)
#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <gc.h>

int main(int argc, char **argv) {
  GC_INIT();
  return Catch::Session().run(argc, argv);
}
