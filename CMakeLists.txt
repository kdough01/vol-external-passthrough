cmake_minimum_required(VERSION 3.18)

option(USE_CUDA "Enable CUDA GPU compression" OFF)

if(USE_CUDA)
  project(HDF5_VOL_PASSTHROUGH VERSION 0.0.1 LANGUAGES C CUDA)
else()
  project(HDF5_VOL_PASSTHROUGH VERSION 0.0.1 LANGUAGES C)
endif()

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

find_package(HDF5 REQUIRED)
find_package(LibPressio REQUIRED)

# Build the source list, adding CUDA conditionally
set(VOL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/H5VLpassthru_ext.c
    cpu_compress.c
)

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND VOL_SRCS gpu_transform.cu)
endif()

add_library(hdf5_vol_passthrough SHARED ${VOL_SRCS} vol_errors.h)

find_package(MPI)
if(MPI_FOUND)
    target_link_libraries(hdf5_vol_passthrough PRIVATE MPI::MPI_C)
    target_compile_definitions(hdf5_vol_passthrough PRIVATE HAVE_MPI)
endif()

target_compile_definitions(hdf5_vol_passthrough PRIVATE ENABLE_EXT_PASSTHRU_LOGGING)

if(USE_CUDA)
  target_compile_definitions(hdf5_vol_passthrough PRIVATE USE_CUDA)
  target_include_directories(hdf5_vol_passthrough
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(hdf5_vol_passthrough PRIVATE CUDA::cudart)
  set_target_properties(hdf5_vol_passthrough PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(hdf5_vol_passthrough
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC  ${HDF5_INCLUDE_DIRS} ${LIBPRESSIO_INCLUDE_DIRS}
)
target_link_libraries(hdf5_vol_passthrough PRIVATE
  ${HDF5_LIBRARIES}
  LibPressio::libpressio
)

add_subdirectory(tests)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/H5VLpassthru_ext.h
  DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_INCLUDE_DIR}
  COMPONENT headers
)
install(
  TARGETS hdf5_vol_passthrough
  EXPORT  ${HDF5_VOL_PASSTHROUGH_EXPORTED_TARGETS}
  LIBRARY DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR}
)