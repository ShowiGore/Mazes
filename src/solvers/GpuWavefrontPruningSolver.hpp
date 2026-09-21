#ifndef MAZES_GPUWAVEFRONTPRUNINGSOLVER_HPP
#define MAZES_GPUWAVEFRONTPRUNINGSOLVER_HPP

#include "GpuSolver.hpp"
#include <memory>
#include <string>

/**
 * @brief Ultra-scale GPU solver combining asynchronous bitpacked dead-end pruning
 * with dual-frontier bitmapped wavefront expansion on the pruned skeleton.
 */
class GpuWavefrontPruningSolver : public GpuSolver {
private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    int prune_passes = 0;
    size_t total_pruned = 0;
    int expansions_count = 0;
    size_t visited_count = 0;
    int max_prune_iterations = -1; // -1 indicates adaptive determination
    int user_min_prune_threshold = 0; // 0 indicates adaptive threshold

    bool ensureOpenCLInitialized();

public:
    explicit GpuWavefrontPruningSolver(std::string device_vendor = "any");
    ~GpuWavefrontPruningSolver() override;

    GpuWavefrontPruningSolver(GpuWavefrontPruningSolver&&) noexcept;
    GpuWavefrontPruningSolver& operator=(GpuWavefrontPruningSolver&&) noexcept;

    bool solve(const Maze &maze) override;

    [[nodiscard]] int getPrunePasses() const { return prune_passes; }
    [[nodiscard]] size_t getPrunedCount() const { return total_pruned; }
    [[nodiscard]] int getExpansionsCount() const { return expansions_count; }
    [[nodiscard]] size_t getVisitedCount() const { return visited_count; }
    void setMaxPruneIterations(int iters) { max_prune_iterations = iters; }
    void setMinPruneThreshold(int threshold) { user_min_prune_threshold = threshold; }
};

#endif // MAZES_GPUWAVEFRONTPRUNINGSOLVER_HPP
