#ifndef MAZES_GPUDEADENDFILLINGSOLVER_HPP
#define MAZES_GPUDEADENDFILLINGSOLVER_HPP

#include "Solver.hpp"
#include <string>
#include <memory>

/**
 * @brief GPU Dead-End Filling Solver (OpenCL Vendor-Agnostic)
 *
 * Parallel cellular automaton running on OpenCL (NVIDIA, AMD, Intel, Apple, POCL).
 * Iteratively prunes all dead-end branches in parallel until convergence,
 * leaving exclusively the unique solution path between Start and End.
 */
class GpuDeadEndFillingSolver : public Solver {
private:
    std::string preferred_device_vendor;
    std::string selected_platform_name;
    std::string selected_device_name;
    size_t total_iterations = 0;
    bool is_initialized = false;

    // Private implementation struct to avoid leaking OpenCL headers in public API
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    bool ensureOpenCLInitialized();

public:
    explicit GpuDeadEndFillingSolver(std::string device_vendor = "any");
    ~GpuDeadEndFillingSolver() override;

    GpuDeadEndFillingSolver(GpuDeadEndFillingSolver&&) noexcept;
    GpuDeadEndFillingSolver& operator=(GpuDeadEndFillingSolver&&) noexcept;

    // Non-copyable due to OpenCL context handles
    GpuDeadEndFillingSolver(const GpuDeadEndFillingSolver&) = delete;
    GpuDeadEndFillingSolver& operator=(const GpuDeadEndFillingSolver&) = delete;

    [[nodiscard]] bool isGpuAvailable();
    [[nodiscard]] std::string getDeviceName() const { return selected_device_name; }
    [[nodiscard]] std::string getPlatformName() const { return selected_platform_name; }
    [[nodiscard]] size_t getIterationCount() const { return total_iterations; }

    bool solve(const Maze &maze) override;
};

#endif // MAZES_GPUDEADENDFILLINGSOLVER_HPP
