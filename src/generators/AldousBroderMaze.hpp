#ifndef MAZES_ALDOUSBRODERMAZE_HPP
#define MAZES_ALDOUSBRODERMAZE_HPP

#include "Maze.hpp"

/**
 * @brief Aldous-Broder Maze Generator
 *
 * The Aldous-Broder algorithm is an elegant algorithm based on random walks
 * that generates a Uniform Spanning Tree (UST) of a graph with completely
 * unbiased probability distribution (every possible spanning tree has an
 * equal chance of being generated).
 *
 * Algorithm Concept:
 * 1. Choose a random starting cell and mark it as visited (part of the maze).
 * 2. Perform a random walk across the grid:
 *    - At each step, choose a random adjacent neighbor.
 *    - If the neighbor has not been visited yet, carve a passage between the
 *      current cell and that neighbor, mark it as visited, and increment the
 *      count of visited cells.
 *    - Move to the neighbor (regardless of whether it was previously visited).
 * 3. Stop when all cells in the grid have been visited.
 *
 * Characteristics & Performance:
 * - Strengths: Extremely simple, produces unbiased mazes with no directional bias.
 *   Explores new cells very quickly in the early stages.
 * - Coupon Collector's Problem: In the final stages, the random walk spends
 *   significant time wandering through already-visited cells trying to hit the
 *   few remaining unvisited cells.
 *
 * Ultra-Efficient Optimizations implemented here:
 * - Bit-Buffered PRNG: Extracts 2-bit directions (0-3) from a single 32-bit PRNG call,
 *   reducing PRNG calls by ~94% without modulo bias.
 * - Zero Heap Allocations: No vectors or dynamic structures created inside the walk loop.
 * - Flat 1D Visited Buffer: Cache-friendly contiguous memory layout for O(1) visited checks.
 * - Direct Midpoint Arithmetic: Computes wall coordinates directly without offset tables.
 */
class AldousBroderMaze : public Maze {
private:
    void init() override;
    void generate() override;

public:
    AldousBroderMaze(int height, int width, unsigned int seed);
    AldousBroderMaze(int height, int width);
};

#endif // MAZES_ALDOUSBRODERMAZE_HPP
