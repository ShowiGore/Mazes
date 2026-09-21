#ifndef MAZES_GREEDYBESTFIRSTSOLVER_HPP
#define MAZES_GREEDYBESTFIRSTSOLVER_HPP

#include "Solver.hpp"

/**
 * @brief Greedy Best-First Search Maze Solver (CPU)
 *
 * Memory-efficient heuristic pathfinder optimized for large spanning tree mazes.
 *
 * Key Optimizations over standard A*:
 * - No g_score table: In an unweighted tree maze with a unique path, computing
 *   exact cumulative distances is redundant. Eliminates 40 GB of RAM on 100k x 100k.
 * - Direction-Compressed Parent Tracking: Instead of 8-byte (r, c) coordinate pairs,
 *   we store a 1-byte incoming direction (UP, RIGHT, DOWN, LEFT) in a flat 1D array.
 *   This cuts parent tracking memory by 87.5% (from 80 GB down to 10 GB on 100k x 100k).
 * - Fast Manhattan Heuristic: Explores nodes in order of minimum Manhattan distance
 *   to the destination, finding the goal in minimal steps.
 */
class GreedyBestFirstSolver : public Solver {
public:
    bool solve(const Maze &maze) override;
};

#endif // MAZES_GREEDYBESTFIRSTSOLVER_HPP
