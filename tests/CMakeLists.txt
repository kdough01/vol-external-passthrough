# TESTS

foreach(test_target test_cpu_write test_cpu_hurricane miranda)
    add_executable(${test_target} ${test_target}.c)
    target_include_directories(${test_target} PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
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
target_link_libraries(test_cpu_hurricane PRIVATE m)
target_link_libraries(miranda PRIVATE m)