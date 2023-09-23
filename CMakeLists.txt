cmake_minimum_required(VERSION 3.25)
project(Mazes)

set(CMAKE_CXX_STANDARD 23)

add_executable(Mazes main.cpp main.hpp utilities/TimeProfiler.cpp utilities/TimeProfiler.hpp generators/Maze.cpp generators/Maze.hpp constants.hpp generators/RecursiveDivisionMaze.cpp generators/RecursiveDivisionMaze.hpp)

#set(PNG_COMPILE_FLAGS "`libpng-config --cflags`")
#set(PNG_LINK_FLAGS "`libpng-config --ldflags`")
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${PNG_COMPILE_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${PNG_LINK_FLAGS}")
#set(CMAKE_CXX_COMPILER /usr/bin/g++)
