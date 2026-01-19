cmake_minimum_required(VERSION 3.20)
set(CMAKE_VERBOSE_MAKEFILE on)

project(voxels)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(MATH_LIBS m gsl gslcblas)
#set(PY_LIBS crypt pthread dl util rt python3.9)
#set(PY_LIBS crypt pthread dl util rt python3.8)
set(PY_LIBS crypt pthread dl util rt python3)
# specific python version (e.g. 3.9 but not just 3) is necessary

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#message(STATUS "${Python_EXECUTABLE}")
include_directories($ENV{HOME}/anaconda3/include)
#include_directories($ENV{HOME}/anaconda3/include/python3.8)
link_directories($ENV{HOME}/anaconda3/lib)
link_directories($ENV{HOME}/anaconda3/lib/python3.9/config-3.9-x86_64-linux-gnu)

find_package(pybind11 REQUIRED)
add_executable(voxels main.cpp voxels.hpp voxels.cpp)
target_link_libraries(voxels ${MATH_LIBS} ${PY_LIBS} pybind11::embed)

find_package(pybind11 REQUIRED)
pybind11_add_module(voxels.so voxels.hpp voxels.cpp voxels_py.cpp)
target_link_libraries(voxels.so PRIVATE ${MATH_LIBS} ${PY_LIBS})

#execute_process(COMMAND chmod 666 .idea/workspace.xml)
