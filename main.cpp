#include "main.hpp"
#include <unistd.h>

int main() {
    TimeProfiler tp;
    tp.start();

    srand(time(NULL));

    std::cout << "Generating maze" << std::endl;
    FractalRecursiveDivisionMaze m(HEIGHT, WIDTH, 0);

    //m.print();
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
