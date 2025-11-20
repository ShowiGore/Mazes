#include <iostream>
#include <time.h>
#include <unistd.h>
#include "constants.hpp"
#include "utilities/TimeProfiler.hpp"
#include "generators/Maze.hpp"
#include "generators/RecursiveDivisionMaze.hpp"
#include "generators/FractalRecursiveDivisionMaze.hpp"

int main() {
    TimeProfiler tp;
    tp.start();

    srand(time(NULL));

    std::cout << "Generating maze" << std::endl;
    FractalRecursiveDivisionMaze m(HEIGHT, WIDTH, 0);

    m.print();
    //m.printSimple();
    std::cout << "Solving maze" << std::endl;
    m.solve();
    //m.save_maze();

    std::cout << "Saving solution" << std::endl;
    m.save_solution();

    tp.stop();
    tp.print();

    return 0;
}
//g++ main.cpp utilities/TimeProfiler.cpp generators/Maze.cpp generators/RecursiveDivisionMaze.cpp -o Mazes.o `libpng-config --ldflags` && ./Mazes.o
