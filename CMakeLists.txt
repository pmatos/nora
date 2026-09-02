cmake_minimum_required(VERSION 3.24)

project(nora LANGUAGES C CXX VERSION 102)
include(GNUInstallDirs)
include(CTest)
include(FetchContent)

set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

# The C++ standard whose features are required to build nora.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# The NIR MLIR dialect is experimental and not used by the interpreter yet, so
# MLIR is opt-in. The default build depends only on LLVM + GMP.
option(NORA_ENABLE_MLIR "Build the experimental NIR MLIR dialect (requires MLIR)" OFF)

# Treat compiler warnings as errors by default (CMake >= 3.24). The code is
# warning-clean on GCC and Clang; disable with -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
# for exploratory local builds.
option(NORA_WERROR "Treat compiler warnings as errors" ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ${NORA_WERROR})

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Oz -flto")

# Add new build type COVERAGE
message("* Adding build type COVERAGE...")
SET(CMAKE_CXX_FLAGS_COVERAGE
  "${CMAKE_CXX_FLAGS_DEBUG} --coverage"
  CACHE STRING "Flags used by the C++ compiler during coverage builds."
  FORCE)
SET(CMAKE_C_FLAGS_COVERAGE
  "${CMAKE_C_FLAGS_DEBUG} --coverage"
  CACHE STRING "Flags used by the C compiler during coverage builds."
  FORCE)
SET(CMAKE_EXE_LINKER_FLAGS_COVERAGE
  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} --coverage"
  CACHE STRING "Flags used for linking binaries during coverage builds."
  FORCE)
SET(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
  "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}"
  CACHE STRING "Flags used by the shared libraries linker during coverage builds."
  FORCE)
MARK_AS_ADVANCED(
  CMAKE_CXX_FLAGS_COVERAGE
  CMAKE_C_FLAGS_COVERAGE
  CMAKE_EXE_LINKER_FLAGS_COVERAGE
  CMAKE_SHARED_LINKER_FLAGS_COVERAGE)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug
    CACHE STRING "Choose the type of build : None Debug Release RelWithDebInfo MinSizeRel Coverage."
    FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

message("* Current build type is : ${CMAKE_BUILD_TYPE}")

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# For git users, attempt to generate a more useful version string
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git QUIET REQUIRED)
  execute_process(COMMAND
    "${GIT_EXECUTABLE}" --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git describe --always HEAD
    RESULT_VARIABLE
    GIT_VERSION_RESULT
    OUTPUT_VARIABLE
    GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(${GIT_VERSION_RESULT})
    message(WARNING "Error running git describe to determine version")
  else()
    set(PROJECT_VERSION "${PROJECT_VERSION} (${GIT_VERSION})")
  endif()
endif()

configure_file(config.h.in src/include/config.h)

# Look for Python3 for lit
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Configure GMP
find_library(gmp gmp)

if(NOT gmp)
  message(FATAL_ERROR "GMP library not found!")
endif()

message(STATUS "Found GMP library")

# Configure the Boehm-Demers-Weiser GC (M2 value model). Shipped as pkg-config
# `bdw-gc`; exposes the imported target PkgConfig::BDWGC.
find_package(PkgConfig REQUIRED)
pkg_check_modules(BDWGC REQUIRED IMPORTED_TARGET bdw-gc)
message(STATUS "Found Boehm GC ${BDWGC_VERSION}")

# Configure LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

if(NORA_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  include(AddMLIR)
  include_directories(${MLIR_INCLUDE_DIRS})
endif()

# Configure src
add_subdirectory(src)

# Configure lit tests
add_subdirectory(test)

# FIXME: From cmake 3.22 or so, we can set permissions with the configure_file command, which is more straightforward.
configure_file(scripts/nora-lit.in ${CMAKE_BINARY_DIR}/nora-lit @ONLY)
file(COPY ${CMAKE_BINARY_DIR}/nora-lit
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
  DESTINATION ${CMAKE_BINARY_DIR}/bin
)
file(REMOVE ${CMAKE_BINARY_DIR}/nora-lit)
