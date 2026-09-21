#ifndef MAZES_GPUBIDIRECTIONALBFSSOLVER_HPP
#define MAZES_GPUBIDIRECTIONALBFSSOLVER_HPP

#include "Solver.hpp"
#include <string>
#include <memory>

/**
 * @brief GPU Parallel Frontier Expansion Solver (Bidirectional BFS)
 *
 * Expands only the active frontier nodes in parallel on the GPU rather than
 * re-scanning the entire grid. Achieves orders-of-magnitude faster convergence
 * on massive spanning tree mazes.
 */
class GpuBidirectionalBfsSolver : public Solver {
private:
    std::string preferred_device_vendor;
    std::string selected_platform_name;
    std::string selected_device_name;
    size_t total_frontier_expansions = 0;
    size_t cells_visited = 0;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    bool ensureOpenCLInitialized();

public:
    explicit GpuBidirectionalBfsSolver(std::string device_vendor = "any");
    ~GpuBidirectionalBfsSolver() override;

    GpuBidirectionalBfsSolver(GpuBidirectionalBfsSolver&&) noexcept;
    GpuBidirectionalBfsSolver& operator=(GpuBidirectionalBfsSolver&&) noexcept;

    GpuBidirectionalBfsSolver(const GpuBidirectionalBfsSolver&) = delete;
    GpuBidirectionalBfsSolver& operator=(const GpuBidirectionalBfsSolver&) = delete;

    [[nodiscard]] std::string getDeviceName() const { return selected_device_name; }
    [[nodiscard]] std::string getPlatformName() const { return selected_platform_name; }
    [[nodiscard]] size_t getVisitedCount() const { return cells_visited; }
    [[nodiscard]] size_t getExpansionsCount() const { return total_frontier_expansions; }

    bool solve(const Maze &maze) override;
};

#endif // MAZES_GPUBIDIRECTIONALBFSSOLVER_HPP
