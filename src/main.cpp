#include <iostream>
#include <time.h>
#include <unistd.h>
#include "utilities/TimeProfiler.hpp"
#include "generators/Maze.hpp"
#include "generators/RecursiveDivisionMaze.hpp"
#include "generators/FractalRecursiveDivisionMaze.hpp"
#include "generators/WilsonsMaze.hpp"

int main() {
    constexpr int HEIGHT = 12345; //524288
    constexpr int WIDTH = 12345; //524288
    //std::random_device rd;
    //const unsigned int seed = rd();
    constexpr unsigned int seed = 0;

    TimeProfiler tp;

    std::cout << "Generating maze" << std::endl;

    tp.start();
    WilsonsMaze m(HEIGHT, WIDTH, seed);
    tp.stop();
    tp.print();

    //m.print();
    //m.printSimple();
    std::cout << "Saving maze" << std::endl;
    m.save_maze();

    std::cout << "Solving maze" << std::endl;
    tp.start();
    m.solve();
    tp.stop();
    tp.print();
    //m.save_maze();

    std::cout << "Saving solution" << std::endl;
    m.save_solution();

    return 0;
}
//g++ main.cpp utilities/TimeProfiler.cpp generators/Maze.cpp generators/RecursiveDivisionMaze.cpp -o Mazes.o `libpng-config --ldflags` && ./Mazes.o
