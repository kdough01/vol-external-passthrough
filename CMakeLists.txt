cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA           "Enable CUDA GPU compression" OFF)
option(BUILD_TESTS        "Build the test suite"        ON)
option(BUILD_EXAMPLES     "Build the example programs"  ON)

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
if(USE_CUDA)
  project(HDF5_VOL_PASSTHROUGH VERSION 0.0.1 LANGUAGES C CXX CUDA)
else()
  project(HDF5_VOL_PASSTHROUGH VERSION 0.0.1 LANGUAGES C CXX)
endif()

# Put build artifacts in predictable locations.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Install directory layout
# ---------------------------------------------------------------------------
if(NOT HDF5_VOL_PASSTHROUGH_INSTALL_BIN_DIR)
  set(HDF5_VOL_PASSTHROUGH_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()
if(NOT HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR)
  set(HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(NOT HDF5_VOL_PASSTHROUGH_INSTALL_INCLUDE_DIR)
  set(HDF5_VOL_PASSTHROUGH_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(NOT HDF5_VOL_PASSTHROUGH_INSTALL_DATA_DIR)
  set(HDF5_VOL_PASSTHROUGH_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

# ---------------------------------------------------------------------------
# Dependencies (resolved once, shared by all subdirectories)
# ---------------------------------------------------------------------------
find_package(HDF5      REQUIRED)
find_package(LibPressio REQUIRED)
find_package(MPI)

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(src)

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()