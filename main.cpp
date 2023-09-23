#include "main.hpp"
#include <unistd.h>

int main() {
    TimeProfiler tp;
    tp.start();

    srand(time(NULL));

    RecursiveDivisionMaze m(HEIGHT, WIDTH);

    m.print();
    //m.printSimple();
    //m.png();

    tp.stop();
    tp.print();

    return 0;
}
//g++ main.cpp utilities/TimeProfiler.cpp generators/Maze.cpp generators/RecursiveDivisionMaze.cpp -o Mazes.o `libpng-config --ldflags` && ./Mazes.o
