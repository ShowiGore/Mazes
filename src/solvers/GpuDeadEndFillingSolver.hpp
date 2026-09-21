#ifndef MAZES_GPUDEADENDFILLINGSOLVER_HPP
#define MAZES_GPUDEADENDFILLINGSOLVER_HPP

#include "Solver.hpp"
#include <string>
#include <memory>

enum class GpuDeadEndMode {
    BITPACKED,    // 32-cell SIMD bitplanes (lowest VRAM & bandwidth, fastest)
    SUBSTEPPING,  // Shared memory halo R=2 (2 steps per global VRAM roundtrip)
    COALESCED     // Coalesced 32x8 1-step kernel
};

/**
 * @brief GPU Dead-End Filling Solver (OpenCL Vendor-Agnostic)
 *
 * Supports three execution modes:
 * - BITPACKED: 32-cell SIMD words using parallel bitplane full-adders.
 * - SUBSTEPPING: Multi-step cellular automaton in shared memory (halo R=2).
 * - COALESCED: Natural 32x8 warp-coalesced 1-step kernel.
 */
class GpuDeadEndFillingSolver : public Solver {
private:
    std::string preferred_device_vendor;
    std::string selected_platform_name;
    std::string selected_device_name;
    size_t total_iterations = 0;
    bool is_initialized = false;
    GpuDeadEndMode mode = GpuDeadEndMode::BITPACKED;
    int user_batch_size = -1;
    std::string config_rationale;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    bool ensureOpenCLInitialized();

public:
    explicit GpuDeadEndFillingSolver(std::string device_vendor = "any");
    ~GpuDeadEndFillingSolver() override;

    GpuDeadEndFillingSolver(GpuDeadEndFillingSolver&&) noexcept;
    GpuDeadEndFillingSolver& operator=(GpuDeadEndFillingSolver&&) noexcept;

    GpuDeadEndFillingSolver(const GpuDeadEndFillingSolver&) = delete;
    GpuDeadEndFillingSolver& operator=(const GpuDeadEndFillingSolver&) = delete;

    [[nodiscard]] bool isGpuAvailable();
    [[nodiscard]] std::string getDeviceName() const { return selected_device_name; }
    [[nodiscard]] std::string getPlatformName() const { return selected_platform_name; }
    [[nodiscard]] size_t getIterationCount() const { return total_iterations; }
    [[nodiscard]] std::string getConfigRationale() const { return config_rationale; }

    void setMode(GpuDeadEndMode new_mode) { mode = new_mode; }
    [[nodiscard]] GpuDeadEndMode getMode() const { return mode; }

    void setBatchSize(int batch) { user_batch_size = batch; }
    [[nodiscard]] int getBatchSize() const { return user_batch_size; }

    bool solve(const Maze &maze) override;
};

#endif // MAZES_GPUDEADENDFILLINGSOLVER_HPP
