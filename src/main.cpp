#include <iostream>
#include "generators/HoustonMaze.hpp"
#include "utilities/TimeProfiler.hpp"
#include "solvers/BidirectionalGreedyBestFirstSolver.hpp"

int main() {
    constexpr int HEIGHT = 10001;
    constexpr int WIDTH = 10001;
    constexpr unsigned int seed = 42;

    TimeProfiler tp;

    std::cout << "========================================" << std::endl;
    std::cout << "Generating Houston Maze (" << HEIGHT << "x" << WIDTH << ")" << std::endl;
    std::cout << "Total grid cells: " << static_cast<size_t>(HEIGHT) * WIDTH << " (~10.00 Billion cells)" << std::endl;
    std::cout << "Graph nodes: " << (static_cast<size_t>(HEIGHT) / 2) * (WIDTH / 2) << " (~2.50 Billion nodes)" << std::endl;
    std::cout << "========================================" << std::endl;

    tp.start();
    HoustonMaze maze(HEIGHT, WIDTH, seed);
    tp.stop();
    std::cout << "Generation time: ";
    tp.print();

    std::cout << "Saving maze image..." << std::endl;
    maze.save_maze();

    std::cout << "\nSolving maze with Bidirectional Greedy Best-First Search (CPU)..." << std::endl;
    BidirectionalGreedyBestFirstSolver solver;
    tp.start();
    const bool solvable = solver.solve(maze);
    tp.stop();
    std::cout << "Bidirectional Greedy BFS solve time: ";
    tp.print();
    std::cout << "Cells visited: " << solver.getVisitedCount() << std::endl;
    std::cout << "Solvable: " << (solvable ? "YES" : "NO") << std::endl;

    if (solvable) {
        std::cout << "Saving solution image..." << std::endl;
        solver.save_solution(maze);
    } else {
        std::cout << "Failed to solve the maze!" << std::endl;
    }

    std::cout << "All operations completed successfully!" << std::endl;
    return 0;
}
