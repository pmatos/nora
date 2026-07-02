include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

add_subdirectory(include)

set(LIBS gmp)

if(NORA_ENABLE_MLIR)
        add_subdirectory(mlir)
        get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
        get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
        list(APPEND LIBS
                ${dialect_libs}
                ${conversion_libs}
                MLIROptLib
                nirLib
        )
endif()

# norac uses LLVM Support APIs directly (cl, InitLLVM, MemoryBuffer,
# raw_ostream). Link the Support component explicitly so the default
# (MLIR-off) build also links on LLVM installs without the libLLVM dylib.
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(norac
        AnalysisFreeVars.cpp
        AST.cpp
        ASTRuntime.cpp
        Diagnostics.cpp
        Environment.cpp
        IdPool.cpp
        Interpreter.cpp
        Lex.cpp
        main.cpp
        Parse.cpp
        Runtime.cpp
        SourceStream.cpp
        UTF8.cpp
)
target_link_libraries(norac PRIVATE ${LIBS})

# Enable sanitizers
if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
        if(WITH_UBSAN OR WITH_ASAN)
                message(FATAL_ERROR "Undefined behavior sanitizer and address sanitizer can only be enabled in Debug mode.")
        endif()
else()
        if(WITH_UBSAN)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
                target_link_options(norac BEFORE PUBLIC -fsanitize=undefined)
        endif()

        if(WITH_ASAN)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
                target_link_options(norac BEFORE PUBLIC -fsanitize=address)
        endif()
endif()
