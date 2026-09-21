#ifndef MAZES_WILSONSMAZE_HPP
#define MAZES_WILSONSMAZE_HPP

#include "Maze.hpp"

/**
 * @brief Wilson's Maze Generator (Loop-Erased Random Walk)
 *
 * Wilson's algorithm generates a Uniform Spanning Tree (UST) of a graph with
 * mathematically proven unbiased probability distribution, using Loop-Erased
 * Random Walks (LERW).
 *
 * Algorithm Concept:
 * 1. Choose an arbitrary cell and mark it as part of the tree.
 * 2. While there are cells not yet in the tree:
 *    a. Pick an unvisited cell and begin a random walk.
 *    b. At each step, record the direction taken: next_dir[curr] = dir.
 *       If the walk loops back and revisits a cell, the direction is simply
 *       overwritten. This implicitly and instantly erases any loops in O(1)!
 *    c. Continue walking until the walk hits ANY cell already in the tree.
 *    d. Retrace the path from the start of the walk to the tree, carving
 *       passages through walls and adding all traversed cells into the tree.
 * 3. By Wilson's 1996 theorem, the order in which starting cells are chosen
 *    does not affect the uniform distribution of the resulting spanning tree.
 *
 * Characteristics & Performance:
 * - Unlike Aldous-Broder, Wilson's algorithm does not suffer from the Coupon
 *   Collector's problem at the end. As more cells join the tree, random walks
 *   terminate faster and faster!
 * - Expected running time on a 2D grid is proportional to the mean commute time:
 *   O(N log N) where N is the number of cells.
 *
 * Ultra-Efficient Optimizations implemented here:
 * - Zero Dynamic Allocations: No vectors constructed or pushed to inside the walk.
 * - Flat 1D Contiguous Memory: in_tree and next_dir use flat arrays for peak cache locality.
 * - Bit-Buffered PRNG: 2-bit direction extraction yields 16 directions per 32-bit PRNG call.
 * - Sequential Scanning: Wilson's theorem guarantees uniformity regardless of walk root order,
 *   eliminating the need to shuffle millions of coordinates.
 */
class WilsonsMaze : public Maze {
private:
    void init() override;
    void generate() override;

public:
    WilsonsMaze(int height, int width, unsigned int seed);
    WilsonsMaze(int height, int width);
};

#endif // MAZES_WILSONSMAZE_HPP
