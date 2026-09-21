#ifndef MAZES_UNBIASEDELLERMAZE_HPP
#define MAZES_UNBIASEDELLERMAZE_HPP

#include "Maze.hpp"
#include <vector>
#include <cstdint>

/**
 * =============================================================================
 * UNBIASED / ISOTROPIC ELLER'S MAZE GENERATOR
 * =============================================================================
 *
 * 1. ALGORITHM OVERVIEW:
 *    - Generates a perfect spanning-tree maze row-by-row in O(W) memory.
 *    - Memory consumption is independent of height H: for any arbitrary height,
 *      only the current and next row states are held in memory.
 *    - Time complexity is strictly linear O(H * W), generating millions of cells
 *      per second with near-zero memory footprint.
 *
 * 2. THEORETICAL BIAS MITIGATION:
 *    Classic Eller's algorithm (1982) exhibits three well-documented structural biases:
 *    a) Singleton Vertical Bias (P_down(1) = 1.0):
 *       Every set must send >= 1 vertical drop. Classic Eller forces 100% of singletons
 *       to drop down, creating severe vertical corridor dominance over horizontal.
 *       -> MITIGATION: For any set of size k >= 2, we pick 1 drop uniformly at random,
 *          and each remaining cell drops with calibrated probability:
 *              p_extra(k) = max(0.0, min(0.5, (k / 2.0 - 1.0) / (k - 1.0)))
 *          This mathematically guarantees that E[drops | k] = k / 2 for all k >= 2,
 *          matching the exact 50% horizontal / 50% vertical passage symmetry of a 2D UST!
 *
 *    b) Bottom-Row "Highway" Collapse:
 *       Classic Eller forcibly merges all disjoint sets on the final row, creating
 *       an artificial, uninterrupted horizontal corridor along the bottom edge.
 *       -> MITIGATION: Progressive Hierarchical Convergence over the final K rows:
 *          p_h(r) increases smoothly over the last K rows (K ~ min(16, max(4, R / 8))):
 *              p_h(r) = 0.5 + 0.5 * ((r - (R - K)) / K)^2
 *          By the time the final row is reached, only a handful of disjoint sets remain,
 *          eliminating the unnatural bottom-row highway.
 *
 *    c) Directional Left-to-Right Skew:
 *       Scanning candidate horizontal walls sequentially from left to right introduces
 *       a rightward set-growth skew.
 *       -> MITIGATION: Horizontal merge candidates are evaluated in randomized order
 *          (1D Kruskal pass) using a fast bit-buffered PRNG.
 * =============================================================================
 */
class UnbiasedEllerMaze : public Maze {
private:
    void init() override;
    void generate() override;

public:
    UnbiasedEllerMaze(int height, int width, unsigned int seed);
    UnbiasedEllerMaze(int height, int width);
};

#endif // MAZES_UNBIASEDELLERMAZE_HPP
