cmake_minimum_required(VERSION 3.25)
project(Mazes)

set(CMAKE_CXX_STANDARD 23)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

add_executable(Mazes main.cpp main.hpp utilities/TimeProfiler.cpp utilities/TimeProfiler.hpp generators/Maze.cpp generators/Maze.hpp constants.hpp generators/RecursiveDivisionMaze.cpp generators/RecursiveDivisionMaze.hpp generators/FractalRecursiveDivisionMaze.cpp)

target_link_libraries(Mazes ${PNG_LIBRARIES})


