# Version set according the the cmake versions available in Ubuntu/Bionic:
# https://packages.ubuntu.com/bionic/cmake
cmake_minimum_required(VERSION 3.10.2)

project(nora LANGUAGES C CXX VERSION 102)
include(GNUInstallDirs)

# The C++ standard whose features are required to build nora.
# Keep in sync with scripts/test/shared.py cxx_standard
set(CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# For git users, attempt to generate a more useful version string
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  find_package(Git QUIET REQUIRED)
  execute_process(COMMAND
               "${GIT_EXECUTABLE}" --git-dir=${CMAKE_CURRENT_SOURCE_DIR}/.git describe --tags --match version_*
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

configure_file(config.h.in config.h)

# Look for racket and raco
find_program (RACKET_BOOT racket REQUIRED)
configure_file(scripts/norac.in ${CMAKE_BINARY_DIR}/bin/norac @ONLY)

# Look for Python3 for lit
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Configure lit tests
add_subdirectory(test)

# FIXME: From cmake 3.22 or so, we can set permissions with the configure_file command, which is more straightforward.
configure_file(scripts/nora-lit.in ${CMAKE_BINARY_DIR}/nora-lit @ONLY)
file(COPY ${CMAKE_BINARY_DIR}/nora-lit
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
  DESTINATION ${CMAKE_BINARY_DIR}/bin
  )
file(REMOVE ${CMAKE_BINARY_DIR}/nora-lit)

