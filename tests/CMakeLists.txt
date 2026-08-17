set(VOL_TEST_TARGETS
    test_cpu_write
    test_time_hurricane
    test_time_miranda
    test_ci
    test_error_miranda
    bench_vol_timing
    bench_pressio_timing
    bench_filter_timing
)

foreach(test_target ${VOL_TEST_TARGETS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${test_target}.cc)
        set(test_src ${test_target}.cc)
    else()
        set(test_src ${test_target}.c)
    endif()

    add_executable(${test_target} ${test_src})

    target_include_directories(${test_target} PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${test_target} PRIVATE
        ${HDF5_LIBRARIES}
        MPI::MPI_C
        hdf5_vol_passthrough
        LibPressio::libpressio
        m
    )

    target_link_options(${test_target} PRIVATE
        -Wl,--allow-shlib-undefined
    )
endforeach()

find_path(SZ3_H5Z_INCLUDE_DIR
    NAMES hdf5_sz3/H5Z_SZ3.hpp
    HINTS ${SZ3_PREFIX} ENV SZ3_PREFIX ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include)

find_library(SZ3_H5Z_LIBRARY
    NAMES hdf5sz3
    HINTS ${SZ3_PREFIX} ENV SZ3_PREFIX ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib lib64)

if(SZ3_H5Z_INCLUDE_DIR AND SZ3_H5Z_LIBRARY)
    message(STATUS "H5Z-SZ3 headers: ${SZ3_H5Z_INCLUDE_DIR}")
    message(STATUS "H5Z-SZ3 library: ${SZ3_H5Z_LIBRARY}")
    target_include_directories(bench_filter_timing PRIVATE ${SZ3_H5Z_INCLUDE_DIR})
    target_link_libraries(bench_filter_timing PRIVATE ${SZ3_H5Z_LIBRARY})
    target_compile_definitions(bench_filter_timing PRIVATE VOL_HAVE_H5Z_SZ3=1)
else()
    message(WARNING
        "H5Z-SZ3 not found (need hdf5_sz3/H5Z_SZ3.hpp and libhdf5sz3). "
        "--sz3 will run at DEFAULT settings and its rows will be invalid. "
        "Build sz3 with +hdf5, or pass -DSZ3_PREFIX=<prefix>.")
endif()

find_path(H5ZZFP_INCLUDE_DIR
    NAMES H5Zzfp_plugin.h
    HINTS ${H5ZZFP_PREFIX} ENV H5ZZFP_PREFIX ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES include)

if(H5ZZFP_INCLUDE_DIR)
    message(STATUS "H5Z-ZFP headers: ${H5ZZFP_INCLUDE_DIR}")
    target_include_directories(bench_filter_timing PRIVATE ${H5ZZFP_INCLUDE_DIR})
else()
    message(WARNING
        "H5Zzfp_plugin.h not found -- --zfp will run at DEFAULT settings. "
        "Pass -DH5ZZFP_PREFIX=<prefix> if h5z-zfp is installed elsewhere.")
endif()

set_property(TARGET bench_filter_timing PROPERTY CXX_STANDARD 17)
set_property(TARGET bench_filter_timing PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_options(bench_filter_timing PRIVATE -rdynamic)

add_executable(dump_opts dump_opts.cc)
target_link_libraries(dump_opts PRIVATE LibPressio::libpressio)

if(USE_CUDA)
    foreach(cuda_target bench_pressio_timing bench_vol_timing)
        target_include_directories(${cuda_target} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${cuda_target} PRIVATE CUDA::cudart)
    endforeach()
    target_compile_definitions(bench_pressio_timing PRIVATE BENCH_TIMING_WITH_CUDA)
    target_compile_definitions(bench_vol_timing     PRIVATE USE_CUDA)
endif()