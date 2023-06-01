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
        Parse.cpp
        Lex.cpp
        SourceStream.cpp
        interpreter.cpp
        runtime.cpp
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
