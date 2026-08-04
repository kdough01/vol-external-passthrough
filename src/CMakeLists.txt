# ---------------------------------------------------------------------------
# hdf5_vol_passthrough : the VOL connector shared library
# ---------------------------------------------------------------------------

set(VOL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/H5VLpassthru_ext.c
    ${CMAKE_CURRENT_SOURCE_DIR}/new_h5api.c
    ${CMAKE_CURRENT_SOURCE_DIR}/compress/compress.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/compress/vol_shared_meta.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/compress/vol_progressive.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/compress/vol_container_read.cc
)

add_library(hdf5_vol_passthrough SHARED ${VOL_SRCS})

# ---- Language / feature requirements ----
target_compile_features(hdf5_vol_passthrough PRIVATE cxx_std_17)

# Uncomment to enable verbose passthrough logging:
# target_compile_definitions(hdf5_vol_passthrough PRIVATE ENABLE_EXT_PASSTHRU_LOGGING)

# ---- MPI (optional) ----
if(MPI_FOUND)
  target_link_libraries(hdf5_vol_passthrough PRIVATE MPI::MPI_C)
  target_compile_definitions(hdf5_vol_passthrough PRIVATE HAVE_MPI)
endif()

# ---- CUDA (optional) ----
if(USE_CUDA)
  target_compile_definitions(hdf5_vol_passthrough PRIVATE USE_CUDA)
  target_include_directories(hdf5_vol_passthrough PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(hdf5_vol_passthrough PRIVATE CUDA::cudart)
  set_target_properties(hdf5_vol_passthrough PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---- Include paths ----
target_include_directories(hdf5_vol_passthrough
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compress
  PUBLIC
    ${HDF5_INCLUDE_DIRS}
    ${LIBPRESSIO_INCLUDE_DIRS}
)

# ---- Link dependencies ----
target_link_libraries(hdf5_vol_passthrough PRIVATE
  ${HDF5_LIBRARIES}
  LibPressio::libpressio
)

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/H5VLpassthru_ext.h
  DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_INCLUDE_DIR}
  COMPONENT headers
)

install(
  TARGETS hdf5_vol_passthrough
  LIBRARY DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${HDF5_VOL_PASSTHROUGH_INSTALL_BIN_DIR}
)