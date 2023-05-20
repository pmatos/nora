# Version set according the the cmake versions available in Ubuntu/Bionic:
# https://packages.ubuntu.com/bionic/cmake
cmake_minimum_required(VERSION 3.10.2)

project(nora LANGUAGES C CXX VERSION 102)
include(GNUInstallDirs)
include(CTest)
include(FetchContent)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 NEW)
endif()

# The C++ standard whose features are required to build nora.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto -Werror")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Oz -flto -Werror")

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# FIXME: there are some warnings in GCC - so disable Werror
#        for now.
# From cmake 2.24 onwards we can use CMAKE_COMPILE_WARNING_AS_ERROR
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(FLAGS_TO_REMOVE "-Werror")
  foreach(TYPE "" "_DEBUG" "_RELEASE" "_RELWITHDEBINFO" "_MINSIZEREL")
    set(CURRENT_FLAG_VAR "CMAKE_CXX_FLAGS${TYPE}")
    if(${CURRENT_FLAG_VAR} MATCHES ${FLAGS_TO_REMOVE})
        string(REPLACE ${FLAGS_TO_REMOVE} "" ${CURRENT_FLAG_VAR} "${${CURRENT_FLAG_VAR}}")
    endif()
  endforeach()
endif()

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

# Find and Configure ICU
find_package(ICU REQUIRED COMPONENTS uc io)

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
