#ifndef MAZES_GPUHIERARCHICALPATHFINDINGSOLVER_HPP
#define MAZES_GPUHIERARCHICALPATHFINDINGSOLVER_HPP

#include "GpuSolver.hpp"
#include <string>
#include <memory>

/**
 * @brief GPU Hierarchical Pathfinding (HPA*) Solver
 *
 * Decomposes an H x W grid into 32x32 tiles.
 * Identifies inter-tile portals in parallel on GPU, constructs a compact
 * abstract graph (orders of magnitude smaller than the raw grid), solves
 * the macro-path in milliseconds, and refines internal tile paths in parallel.
 */
class GpuHierarchicalPathfindingSolver : public GpuSolver {
private:
    size_t total_portals = 0;
    size_t macro_steps = 0;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    bool ensureOpenCLInitialized();

public:
    explicit GpuHierarchicalPathfindingSolver(std::string device_vendor = "any");
    ~GpuHierarchicalPathfindingSolver() override;

    GpuHierarchicalPathfindingSolver(GpuHierarchicalPathfindingSolver&&) noexcept;
    GpuHierarchicalPathfindingSolver& operator=(GpuHierarchicalPathfindingSolver&&) noexcept;

    GpuHierarchicalPathfindingSolver(const GpuHierarchicalPathfindingSolver&) = delete;
    GpuHierarchicalPathfindingSolver& operator=(const GpuHierarchicalPathfindingSolver&) = delete;

    [[nodiscard]] size_t getPortalsCount() const { return total_portals; }
    [[nodiscard]] size_t getMacroSteps() const { return macro_steps; }

    bool solve(const Maze &maze) override;
};

#endif // MAZES_GPUHIERARCHICALPATHFINDINGSOLVER_HPP
