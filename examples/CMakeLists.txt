set(${LIBRARY_TARGET_NAME}_TEST
    main.cpp
)

foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} GPU_Mergesort ${CUDA_LIBRARIES})
    target_link_libraries(${testname} GPU_Mergesort cuda_obj)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
