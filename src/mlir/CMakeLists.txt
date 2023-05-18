add_mlir_dialect_library(nirLib
    MLIRGen.cpp
    Dialect.cpp

    ADDITIONAL_HEADER_DIRS 
    ${PROJECT_SOURCE_DIR}/include/nir

    DEPENDS 
    NirIncGen)