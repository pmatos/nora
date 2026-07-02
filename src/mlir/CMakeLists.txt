add_mlir_dialect_library(nirLib
    nora.cpp
    Dialect.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/nir

    DEPENDS
    NirIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

# R5 / B0: a minimal nir-opt driver that registers the NIR dialect so .mlir
# files can be parsed, verified and printed (round-trip). Built only when
# NORA_ENABLE_MLIR is ON (this whole directory is added under that guard).
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(nir-opt nir-opt.cpp)
llvm_update_compile_flags(nir-opt)
target_link_libraries(nir-opt
    PRIVATE
    nirLib
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRSupport
)
mlir_check_all_link_libraries(nir-opt)
