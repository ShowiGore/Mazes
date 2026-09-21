#include <iostream>
#include <unistd.h>
#include "utilities/TimeProfiler.hpp"
#include "generators/Maze.hpp"
#include "generators/RecursiveDivisionMaze.hpp"
#include "generators/FractalRecursiveDivisionMaze.hpp"
#include "generators/WilsonsMaze.hpp"
#include "generators/AldousBroderMaze.hpp"
#include "generators/HoustonMaze.hpp"
#include "solvers/RecursiveSolver.hpp"
#include "solvers/AStarSolver.hpp"

int main() {
    constexpr int HEIGHT = 1001;
    constexpr int WIDTH = 1001;
    constexpr unsigned int seed = 42;

    TimeProfiler tp;
    AStarSolver solver;

    std::cout << "========================================" << std::endl;
    std::cout << "1. Testing Aldous-Broder Maze Generator" << std::endl;
    std::cout << "========================================" << std::endl;

    tp.start();
    AldousBroderMaze ab_maze(HEIGHT, WIDTH, seed);
    tp.stop();
    std::cout << "Aldous-Broder generation time: ";
    tp.print();

    std::cout << "Solving Aldous-Broder maze with A*..." << std::endl;
    tp.start();
    const bool ab_solvable = solver.solve(ab_maze);
    tp.stop();
    std::cout << "A* solve time: ";
    tp.print();
    std::cout << "Solvable: " << (ab_solvable ? "YES" : "NO") << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "2. Testing Wilson's Maze Generator" << std::endl;
    std::cout << "========================================" << std::endl;

    tp.start();
    WilsonsMaze wilson_maze(HEIGHT, WIDTH, seed);
    tp.stop();
    std::cout << "Wilson generation time: ";
    tp.print();

    std::cout << "Solving Wilson's maze with A*..." << std::endl;
    tp.start();
    const bool wilson_solvable = solver.solve(wilson_maze);
    tp.stop();
    std::cout << "A* solve time: ";
    tp.print();
    std::cout << "Solvable: " << (wilson_solvable ? "YES" : "NO") << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "3. Testing Houston's Algorithm (AB + Wilson Hybrid)" << std::endl;
    std::cout << "========================================" << std::endl;

    tp.start();
    HoustonMaze houston_maze(HEIGHT, WIDTH, seed); // Default constructor: automatically uses THEORETICAL_ALPHA (1/3)
    tp.stop();
    std::cout << "Houston (theoretical alpha=1/3) generation time: ";
    tp.print();

    std::cout << "Solving Houston maze with A*..." << std::endl;
    tp.start();
    const bool houston_solvable = solver.solve(houston_maze);
    tp.stop();
    std::cout << "A* solve time: ";
    tp.print();
    std::cout << "Solvable: " << (houston_solvable ? "YES" : "NO") << std::endl;

    std::cout << "\n--- Houston Threshold Transition Sweep (alpha) ---" << std::endl;
    constexpr float thresholds[] = {0.10f, 0.20f, 0.333f, 0.50f, 0.75f};
    for (const float alpha : thresholds) {
        tp.start();
        HoustonMaze sweep_maze(HEIGHT, WIDTH, seed, alpha);
        tp.stop();
        std::cout << "Houston alpha = " << alpha << " : ";
        tp.print();
    }

    std::cout << "\nSaving sample mazes to PNG..." << std::endl;
    houston_maze.save_maze();
    std::cout << "All tests completed successfully!" << std::endl;

    return 0;
}


