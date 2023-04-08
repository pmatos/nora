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

add_executable(test_parse
  test_parse.cpp
  ${PROJECT_SOURCE_DIR}/src/parser/parse.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/arithplus.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/definevalues.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/values.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/identifier.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/integer.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/linklet.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/lambda.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/begin.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/list.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/application.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/setbang.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/ifcond.cpp
  ${PROJECT_SOURCE_DIR}/src/ast/letvalues.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/idpool.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/upcast.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/downcast.cpp
  ${PROJECT_SOURCE_DIR}/src/exprnode.cpp
  ${PROJECT_SOURCE_DIR}/src/valuenode.cpp
)
target_link_libraries(test_parse PRIVATE Catch2::Catch2 gmp gmpxx)

catch_discover_tests(test_parse)