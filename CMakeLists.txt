cmake_minimum_required(VERSION 3.12)
project(HDF5_VOL_PASSTHROUGH VERSION 0.0.1 LANGUAGES C)

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

set(HDF5_VOL_PASSTHROUGH_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/H5VLpassthru_ext.h
)

set(HDF5_VOL_PASSTHROUGH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/H5VLpassthru_ext.c)

find_package(MPI REQUIRED)    
find_package(HDF5)
if (NOT HDF5_FOUND)
  message (FATAL_ERROR "HDF5 was not found on the system")
endif ()
find_package(LibPressio REQUIRED)

add_library(hdf5_vol_passthrough SHARED ${HDF5_VOL_PASSTHROUGH_SRCS}
  cpu_compress.c)
target_compile_definitions(hdf5_vol_passthrough PRIVATE ENABLE_EXT_PASSTHRU_LOGGING)
target_include_directories(hdf5_vol_passthrough 
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PUBLIC "${HDF5_INCLUDE_DIRS}"
)
target_link_libraries(hdf5_vol_passthrough PRIVATE "${HDF5_LIBRARIES}")
target_link_libraries(hdf5_vol_passthrough PRIVATE MPI::MPI_C)
target_link_libraries(hdf5_vol_passthrough PRIVATE LibPressio::libpressio)

add_executable(test_write test_write.c)
target_link_libraries(test_write PRIVATE ${HDF5_LIBRARIES})
target_include_directories(test_write PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_write PRIVATE MPI::MPI_C)

install(
  FILES
    ${HDF5_VOL_PASSTHROUGH_HEADERS}
  DESTINATION
    ${HDF5_VOL_PASSTHROUGH_INSTALL_INCLUDE_DIR}
  COMPONENT
    headers
)

install(
  TARGETS
    hdf5_vol_passthrough
  EXPORT
    ${HDF5_VOL_PASSTHROUGH_EXPORTED_TARGETS}
  LIBRARY DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR}
)
