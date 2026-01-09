#include <iostream>
#include <unistd.h>
#include "utilities/TimeProfiler.hpp"
#include "generators/Maze.hpp"
#include "generators/RecursiveDivisionMaze.hpp"
#include "generators/FractalRecursiveDivisionMaze.hpp"
#include "generators/WilsonsMaze.hpp"
#include "solvers/RecursiveSolver.hpp"
#include "solvers/AStarSolver.hpp"

int main() {
    constexpr int HEIGHT = 12345; //524288
    constexpr int WIDTH = 12345; //524288
    //std::random_device rd;
    //const unsigned int seed = rd();
    constexpr unsigned int seed = 0;

    TimeProfiler tp;

    std::cout << "Generating maze" << std::endl;

    tp.start();
    WilsonsMaze maze(HEIGHT, WIDTH, seed);
    tp.stop();
    tp.print();

    //maze.print();
    //maze.printSimple();
    std::cout << "Saving maze" << std::endl;
    maze.save_maze();

    AStarSolver solver; // Instancia del solver
    solver.setHeuristicWeight(10);

    std::cout << "Solving maze" << std::endl;
    tp.start();
    const bool solvable = solver.solve(maze);
    tp.stop();
    tp.print();

    if (solvable) {
        std::cout << "Saving solution" << std::endl;
        solver.save_solution(maze);
    } else {
        std::cout << "Failed to solve the maze" << std::endl;
    }

    return 0;
}
//g++ main.cpp utilities/TimeProfiler.cpp generators/Maze.cpp generators/RecursiveDivisionMaze.cpp -o Mazes.o `libpng-config --ldflags` && ./Mazes.o
