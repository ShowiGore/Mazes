#ifndef MAZES_GPUUTILS_HPP
#define MAZES_GPUUTILS_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_EXCEPTIONS
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 200
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif
#include <CL/opencl.hpp>

struct GpuAdaptiveConfig {
    int batch_size = 128;
    size_t workgroup_size = 512;
    size_t frontier_capacity = 2000000;
    bool is_auto = true;
    std::string rationale;
};

/**
 * @brief Computes optimal GPU execution parameters adaptively
 *
 * Takes into account:
 * - Maze dimensions (estimated tree diameter / solution depth)
 * - GPU VRAM capacity (memory pressure and headroom)
 * - GPU Compute Units (SMs) and maximum work-group size
 * - Watchdog / TDR safety limits
 *
 * @param device The target OpenCL GPU device
 * @param height Maze height
 * @param width Maze width
 * @param bytes_per_cell Memory footprint per cell of the solver
 * @param user_batch_override Optional user override (-1 for auto)
 */
inline GpuAdaptiveConfig computeGpuAdaptiveConfig(
    const cl::Device &device,
    int height,
    int width,
    size_t bytes_per_cell,
    int user_batch_override = -1
) {
    GpuAdaptiveConfig cfg;

    const cl_ulong vram_bytes = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const cl_uint compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    const size_t max_wg = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // 1. Workgroup size tuning: prefer 512 or 256 (multiples of warp size 32)
    // 512 provides optimal thread occupancy and barrier latency on modern GPUs
    if (max_wg >= 512) {
        cfg.workgroup_size = 512;
    } else if (max_wg >= 256) {
        cfg.workgroup_size = 256;
    } else {
        cfg.workgroup_size = (max_wg / 32) * 32;
        if (cfg.workgroup_size == 0) cfg.workgroup_size = max_wg;
    }

    const size_t total_cells = static_cast<size_t>(height) * width;
    const size_t state_mem_bytes = total_cells * bytes_per_cell;

    // 2. Frontier capacity and memory pressure based on VRAM capacity & headroom
    // Assume up to 6 frontier buffers (ping-pong curr/next for forward & backward, plus secondary queues)
    constexpr size_t NUM_FRONTIER_BUFFERS = 6;
    size_t safe_frontier_cap = 2000000;
    if (vram_bytes > 0) {
        // Reserve 15% headroom for display, driver, and system buffers
        const size_t usable_vram = static_cast<size_t>(vram_bytes * 0.85);
        if (usable_vram > state_mem_bytes) {
            const size_t remaining_for_frontier = usable_vram - state_mem_bytes;
            const size_t max_possible_frontier = remaining_for_frontier / (NUM_FRONTIER_BUFFERS * sizeof(int));
            safe_frontier_cap = std::clamp<size_t>(max_possible_frontier, 250000, 4000000);
        } else {
            safe_frontier_cap = 250000; // minimum fallback under extreme memory pressure
        }
    }
    cfg.frontier_capacity = safe_frontier_cap;

    const size_t total_required_bytes = state_mem_bytes + NUM_FRONTIER_BUFFERS * cfg.frontier_capacity * sizeof(int);
    const double memory_pressure = (vram_bytes > 0)
        ? static_cast<double>(total_required_bytes) / static_cast<double>(vram_bytes)
        : 0.5;

    // 3. Batch size calculation
    if (user_batch_override > 0) {
        cfg.batch_size = std::clamp(user_batch_override, 1, 8192);
        cfg.is_auto = false;
        std::ostringstream oss;
        const double vram_gib = static_cast<double>(vram_bytes) / (1024.0 * 1024.0 * 1024.0);
        oss << "user override (batch=" << cfg.batch_size
            << ", wg=" << cfg.workgroup_size
            << ", frontier_cap=" << (cfg.frontier_capacity / 1000000.0) << "M"
            << " | Maze: " << height << "x" << width
            << ", VRAM: " << std::round(vram_gib * 10.0) / 10.0 << " GiB"
            << ", " << compute_units << " CUs)";
        cfg.rationale = oss.str();
        return cfg;
    }

    // Adaptive batch sizing:
    // a) Maze dimensions and diameter
    const int max_dim = std::max(height, width);
    int base_batch = 128;

    if (max_dim <= 1000) {
        base_batch = 32;
    } else if (max_dim <= 3000) {
        base_batch = 64;
    } else if (max_dim <= 10000) {
        base_batch = 128;
    } else if (max_dim <= 30000) {
        base_batch = 256;
    } else if (max_dim <= 65536) {
        base_batch = 512;
    } else {
        base_batch = 1024;
    }

    // b) Modulation by GPU Compute Units (SMs) and Workgroup Size
    if (compute_units >= 20 && cfg.workgroup_size >= 512 && memory_pressure < 0.40 && base_batch <= 256) {
        base_batch *= 2;
    } else if (compute_units < 16 || cfg.workgroup_size <= 256) {
        base_batch = std::min(base_batch, 128);
    }

    // c) Modulation by Memory Pressure (prevent driver TDR timeouts when VRAM is tight)
    if (memory_pressure > 0.80) {
        base_batch = std::min(base_batch, 64);
    } else if (memory_pressure > 0.65) {
        base_batch = std::min(base_batch, 128);
    }

    cfg.batch_size = std::clamp(base_batch, 16, 2048);
    cfg.is_auto = true;

    std::ostringstream oss;
    const double vram_gib = static_cast<double>(vram_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double total_cells_m = static_cast<double>(total_cells) / 1000000.0;
    oss << "auto-tuned (batch=" << cfg.batch_size
        << ", wg=" << cfg.workgroup_size
        << ", frontier_cap=" << (cfg.frontier_capacity / 1000000.0) << "M"
        << " | Maze: " << height << "x" << width << " [" << std::round(total_cells_m * 10.0) / 10.0 << "M cells]"
        << ", VRAM: " << std::round(vram_gib * 10.0) / 10.0 << " GiB"
        << " [" << std::round(memory_pressure * 1000.0) / 10.0 << "% pressure]"
        << ", " << compute_units << " CUs, max_wg=" << max_wg << ")";
    cfg.rationale = oss.str();

    return cfg;
}

#endif // MAZES_GPUUTILS_HPP
