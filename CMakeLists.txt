project(kdtree)
cmake_minimum_required(VERSION 2.8)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-std=c++17 -lpthread -Wall -O0 -fopenmp")
endif (CMAKE_COMPILER_IS_GNUCXX)

file(GLOB srcs *.cpp *.h*)

add_executable(kdtree ${srcs})
