include_directories(cpp/include)
set(SOURCES Dtw.cpp cpp/src/ColMajorCell.cpp cpp/src/FastDTW.cpp cpp/src/SearchWindow.cpp cpp/src/WarpPath.cpp)
add_library(Dtw SHARED ${SOURCES} )
set_property(TARGET Dtw PROPERTY POSITION_INDEPENDENT_CODE ON)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
set_target_properties(Dtw PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBDIR} RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBDIR} ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LIBDIR})

target_link_libraries(Dtw ${TF_LIB_FILE})
