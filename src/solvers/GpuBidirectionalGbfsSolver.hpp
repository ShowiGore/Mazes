#ifndef MAZES_GPUBIDIRECTIONALGBFSSOLVER_HPP
#define MAZES_GPUBIDIRECTIONALGBFSSOLVER_HPP

#include "GpuSolver.hpp"
#include <string>
#include <memory>

/**
 * @brief GPU Bidirectional Beam-GBFS with Dual-Frontier Bucketing
 *
 * Combines Manhattan heuristic guidance with GPU parallel throughput.
 * Uses a branchless dual-bucket frontier (primary for Delta h <= 0,
 * secondary for Delta h > 0) without maintaining an expensive global priority queue.
 */
class GpuBidirectionalGbfsSolver : public GpuSolver {
private:
    size_t total_frontier_expansions = 0;
    size_t cells_visited = 0;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    bool ensureOpenCLInitialized();

public:
    explicit GpuBidirectionalGbfsSolver(std::string device_vendor = "any");
    ~GpuBidirectionalGbfsSolver() override;

    GpuBidirectionalGbfsSolver(GpuBidirectionalGbfsSolver&&) noexcept;
    GpuBidirectionalGbfsSolver& operator=(GpuBidirectionalGbfsSolver&&) noexcept;

    GpuBidirectionalGbfsSolver(const GpuBidirectionalGbfsSolver&) = delete;
    GpuBidirectionalGbfsSolver& operator=(const GpuBidirectionalGbfsSolver&) = delete;

    [[nodiscard]] size_t getVisitedCount() const { return cells_visited; }
    [[nodiscard]] size_t getExpansionsCount() const { return total_frontier_expansions; }

    bool solve(const Maze &maze) override;
};

#endif // MAZES_GPUBIDIRECTIONALGBFSSOLVER_HPP
