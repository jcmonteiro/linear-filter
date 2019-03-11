cmake_minimum_required(VERSION 2.8)
project(linear-system)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(LIBNAME "linear_system")

find_package(Boost COMPONENTS unit_test_framework)
find_package(yaml-cpp)
find_package(Eigen3 REQUIRED)
find_package(pybind11)

include_directories("include" ${EIGEN3_INCLUDE_DIRS})
add_library(${LIBNAME} SHARED "src/LinearSystem.cpp;src/HelperFunctions.cpp")

# Python bindings
if (pybind11_FOUND)
    pybind11_add_module(${LIBNAME}_py python/python_bindings.cpp)
    target_link_libraries(${LIBNAME}_py PUBLIC ${LIBNAME})
    set(BUILD_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/${LIBNAME}")
    set_target_properties(${LIBNAME}_py
        PROPERTIES
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY "${BUILD_PYTHON_DIR}"
    )
    configure_file(
        "python/__init__.py"
        "${BUILD_PYTHON_DIR}/__init__.py"
    )
    configure_file(
        "python/setup.py"
        "${BUILD_PYTHON_DIR}/setup.py"
    )
endif ()

# Unit tests
if (Boost_FOUND AND yaml-cpp_FOUND)
    ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
    add_executable(test-library "test/test_LinearSystem.cpp")
    target_include_directories(test-library PRIVATE ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
    target_link_libraries(test-library ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES} ${LIBNAME})
    add_test(NAME test-1 COMMAND test-library)
endif ()

# Install cpp library
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "include/LinearSystem.hpp;include/HelperFunctions.hpp")
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

# Install python library
if (pybind11_FOUND)
    install(CODE "execute_process(COMMAND python3 setup.py install
        --prefix=${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ${BUILD_PYTHON_DIR})")
endif ()
