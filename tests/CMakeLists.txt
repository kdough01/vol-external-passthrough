foreach(test_target test_cpu_write test_time_hurricane test_time_miranda test_ci test_error_miranda bench_vol_timing bench_pressio_timing bench_filter_timing)
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
    )
    target_link_options(${test_target} PRIVATE
        -Wl,--allow-shlib-undefined
    )
endforeach()
target_link_libraries(test_time_hurricane       PRIVATE m)
target_link_libraries(test_time_miranda PRIVATE m)
target_link_libraries(test_error_miranda PRIVATE m)
target_link_libraries(test_ci                  PRIVATE m)
target_link_libraries(bench_vol_timing PRIVATE m)
target_link_libraries(bench_pressio_timing PRIVATE m)
target_link_libraries(bench_filter_timing PRIVATE m)

if(USE_CUDA)
  target_compile_definitions(bench_pressio_timing PRIVATE BENCH_TIMING_WITH_CUDA)
  target_include_directories(bench_pressio_timing PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(bench_pressio_timing PRIVATE CUDA::cudart)
endif()