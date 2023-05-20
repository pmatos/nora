Include(FetchContent)

# Use Catch2 for unit testing
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.10)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

include_directories(${CMAKE_SOURCE_DIR}/src/include)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(LLVM_LIBS support)

add_executable(test_parse
  test_parse.cpp
  ${PROJECT_SOURCE_DIR}/src/Lex.cpp
  ${PROJECT_SOURCE_DIR}/src/Parse.cpp
  ${PROJECT_SOURCE_DIR}/src/UTF8.cpp
  ${PROJECT_SOURCE_DIR}/src/SourceStream.cpp
  ${PROJECT_SOURCE_DIR}/src/ast.cpp
  ${PROJECT_SOURCE_DIR}/src/idpool.cpp
  ${PROJECT_SOURCE_DIR}/src/runtime.cpp
)
target_link_libraries(test_parse PRIVATE Catch2::Catch2 gmp gmpxx ${LLVM_LIBS})

catch_discover_tests(test_parse)