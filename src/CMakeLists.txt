# Configure LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MILRConfig.cmake in: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

include_directories(${MLIR_INCLUDE_DIRS})

include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

add_subdirectory(include)
add_subdirectory(mlir)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        gmp
        MLIROptLib
        nirLib
)

add_llvm_executable(norac
        main.cpp
        dumper.cpp
        exprnode.cpp
        valuenode.cpp
        environment.cpp
        utils/idpool.cpp
        utils/upcast.cpp
        parser/parse.cpp
        interpreter/interpreter.cpp
        ast/arithplus.cpp
        ast/definevalues.cpp
        ast/values.cpp
        ast/identifier.cpp
        ast/integer.cpp
        ast/linklet.cpp
        ast/lambda.cpp
        ast/begin.cpp
)
target_link_libraries(norac PRIVATE ${LIBS})