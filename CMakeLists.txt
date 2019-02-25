cmake_minimum_required(VERSION 2.8)
project(linear-system)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(LIBNAME "linear_system")

find_package(Boost COMPONENTS unit_test_framework)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories("include")
add_library(${LIBNAME} SHARED "src/LinearSystem.cpp;src/HelperFunctions.cpp")

# Unit tests
if (Boost_FOUND)
    ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
    add_executable(test-library "test/test_LinearSystem.cpp")
    target_include_directories(test-library PRIVATE ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
    target_link_libraries(test-library ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES} ${LIBNAME})
    add_test(NAME test-1 COMMAND test-library)
endif ()

# Install
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "include/LinearSystem.hpp")
install(
    TARGETS ${LIBNAME}
    EXPORT "${LIBNAME}-export"
    LIBRARY DESTINATION "lib/${LIBNAME}"
    ARCHIVE DESTINATION "lib/${LIBNAME}"
    PUBLIC_HEADER DESTINATION "include/${LIBNAME}"
)
install(
    EXPORT "${LIBNAME}-export"
    FILE "${LIBNAME}-config.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/${LIBNAME}/cmake"
)
