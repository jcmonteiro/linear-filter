cmake_minimum_required(VERSION 3.0)
project(linear-system)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(LIBNAME "linear_system")

if(CMAKE_BUILD_TYPE MATCHES DEBUG OR CMAKE_BUILD_TYPE MATCHES Debug)
    # debug mode stuff here
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og")
else()
    # release mode stuff here
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

use_cxx11()

find_package(GTest QUIET)
find_package(Boost QUIET COMPONENTS unit_test_framework)
find_package(yaml-cpp QUIET)
find_package(Eigen3 REQUIRED)
find_package(pybind11 QUIET)
find_package(PkgConfig QUIET)

include_directories("include" ${EIGEN3_INCLUDE_DIRS})
set(LIBRARY_SOURCES
    src/HelperFunctions.cpp
    src/LinearSystem.cpp
    src/Builder.cpp
)
add_library(${LIBNAME} SHARED "${LIBRARY_SOURCES}")

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
# 1. Boost
OPTION(BOOST_UTF "Create test using Boost Unit Test Framework" OFF) # Disabled by default
if (Boost_UNIT_TEST_FRAMEWORK_FOUND AND yaml-cpp_FOUND AND BOOST_UTF)
    ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
    add_executable(test-library "test/test_LinearSystem.cpp")
    target_include_directories(test-library PRIVATE ${Boost_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
    target_link_libraries(test-library ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES} ${LIBNAME})
    add_test(NAME test-1 COMMAND test-library)
endif ()
# 2. Google Test
OPTION(GTEST_UTF "Create test using Google test Unit Test Framework" OFF) # Disabled by default
if (GTEST_FOUND AND yaml-cpp_FOUND AND GTEST_UTF)
    add_executable(gtest-library "test/gtest_LinearSystem.cpp")
    target_include_directories(gtest-library PRIVATE ${GTEST_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
    target_link_libraries(gtest-library ${GTEST_LIBRARIES} ${YAML_CPP_LIBRARIES} pthread ${LIBNAME})
    #add_test(NAME gtest-1 COMMAND gtest-library)
endif ()

# Install c++ library
if (PkgConfig_FOUND)
    set(PKGCONFIG_REQUIRES "eigen3")
    set(PKGCONFIG_CFLAGS "-std=c++11")
    CONFIGURE_FILE("${CMAKE_CURRENT_LIST_DIR}/cmake/linear-system.pc.in" "linear-system.pc" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/linear-system.pc" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/)
endif ()
set(LIBRARY_HEADERS
    include/LinearSystem.hpp
    include/HelperFunctions.hpp
    include/Builder.hpp
)
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${LIBRARY_HEADERS}")
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
