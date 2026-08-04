set(VOL_EXAMPLE_TARGETS
    new_h5api_ex
    async_new_h5api_ex
)

foreach(example_target ${VOL_EXAMPLE_TARGETS})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_target}.c)
        message(WARNING "examples: ${example_target}.c not found -- skipping")
        continue()
    endif()

    add_executable(${example_target} ${example_target}.c)

    target_include_directories(${example_target} PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${example_target} PRIVATE
        ${HDF5_LIBRARIES}
        hdf5_vol_passthrough
        LibPressio::libpressio
        m
    )

    if(MPI_FOUND)
        target_link_libraries(${example_target} PRIVATE MPI::MPI_C)
    endif()

    target_link_options(${example_target} PRIVATE
        -Wl,--allow-shlib-undefined
    )
endforeach()