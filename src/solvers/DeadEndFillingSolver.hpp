#ifndef MAZES_DEADENDFILLINGSOLVER_HPP
#define MAZES_DEADENDFILLINGSOLVER_HPP

#include "Solver.hpp"

/**
 * @brief Dead-End Filling Maze Solver (CPU)
 *
 * Dead-End Filling is a highly efficient, deterministic O(N) solving algorithm
 * ideally suited for perfect mazes (Uniform Spanning Trees).
 *
 * Mathematical Principle:
 * - A perfect maze contains no cycles (it is a tree graph).
 * - In any tree, every vertex of degree <= 1 (except the designated Start and End)
 *   is a dead end that cannot be part of the simple path connecting Start and End.
 * - Iteratively pruning all dead-end cells causes dead-end branches to collapse
 *   backward toward junctions.
 * - Once no more cells of degree <= 1 exist, the ONLY remaining unpruned path cells
 *   form the exact, unique solution path connecting Start and End!
 *
 * Complexity & Memory:
 * - Time Complexity: O(N) linear time. Every cell is visited a constant number of times.
 * - Space Complexity: O(N) using a flat 1D degree array (1 byte per cell).
 * - Eliminates priority queues, floating-point heuristics, and heavy 2D arrays.
 */
class DeadEndFillingSolver : public Solver {
public:
    bool solve(const Maze &maze) override;
};

#endif // MAZES_DEADENDFILLINGSOLVER_HPP
