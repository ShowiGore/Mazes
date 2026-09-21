#ifndef MAZES_PARALLELPRUNINGDFSSOLVER_HPP
#define MAZES_PARALLELPRUNINGDFSSOLVER_HPP

#include "Solver.hpp"
#include <vector>
#include <cstdint>
#include <string>

/**
 * @brief Ultra-scale multi-threaded dead-end pruning + bidirectional tree-DFS solver.
 *
 * Designed for billions of cells:
 * - Phase 1: OpenMP 64-bit SIMD bitpacked dead-end pruning (Boolean full-adders).
 * - Phase 2: In-place bidirectional DFS with O(1) heap allocation and Manhattan ordering.
 */
class ParallelPruningDfsSolver : public Solver {
private:
    int prune_passes = 0;
    size_t total_pruned = 0;
    size_t visited_count = 0;
    int max_prune_iterations = -1; // -1 indicates adaptive determination
    size_t user_min_prune_threshold = 0; // 0 indicates adaptive threshold

public:
    ParallelPruningDfsSolver();
    ~ParallelPruningDfsSolver() override = default;

    bool solve(const Maze &maze) override;

    [[nodiscard]] int getPrunePasses() const { return prune_passes; }
    [[nodiscard]] size_t getPrunedCount() const { return total_pruned; }
    [[nodiscard]] size_t getVisitedCount() const { return visited_count; }
    void setMaxPruneIterations(int iters) { max_prune_iterations = iters; }
    void setMinPruneThreshold(size_t threshold) { user_min_prune_threshold = threshold; }
};

#endif // MAZES_PARALLELPRUNINGDFSSOLVER_HPP
