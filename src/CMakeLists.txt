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
        environment.cpp
        ast.cpp
        idpool.cpp
        parse.cpp
        interpreter.cpp
        runtime.cpp
)
target_link_libraries(norac PRIVATE ${LIBS})