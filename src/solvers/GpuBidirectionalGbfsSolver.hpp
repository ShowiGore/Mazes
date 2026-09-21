#ifndef MAZES_GPUBIDIRECTIONALGBFSSOLVER_HPP
#define MAZES_GPUBIDIRECTIONALGBFSSOLVER_HPP

#include "Solver.hpp"
#include <string>
#include <memory>

/**
 * @brief GPU Bidirectional Beam-GBFS with Dual-Frontier Bucketing
 *
 * Combines Manhattan heuristic guidance with GPU parallel throughput.
 * Uses a branchless dual-bucket frontier (primary for Delta h <= 0,
 * secondary for Delta h > 0) without maintaining an expensive global priority queue.
 */
class GpuBidirectionalGbfsSolver : public Solver {
private:
    std::string preferred_device_vendor;
    std::string selected_platform_name;
    std::string selected_device_name;
    size_t total_frontier_expansions = 0;
    size_t cells_visited = 0;
    int user_batch_size = -1;
    std::string config_rationale;

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

    [[nodiscard]] std::string getDeviceName() const { return selected_device_name; }
    [[nodiscard]] std::string getPlatformName() const { return selected_platform_name; }
    [[nodiscard]] size_t getVisitedCount() const { return cells_visited; }
    [[nodiscard]] size_t getExpansionsCount() const { return total_frontier_expansions; }
    [[nodiscard]] std::string getConfigRationale() const { return config_rationale; }

    void setBatchSize(int batch) { user_batch_size = batch; }
    [[nodiscard]] int getBatchSize() const { return user_batch_size; }

    bool solve(const Maze &maze) override;
};

#endif // MAZES_GPUBIDIRECTIONALGBFSSOLVER_HPP
