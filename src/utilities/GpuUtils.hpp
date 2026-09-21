#ifndef MAZES_GPUUTILS_HPP
#define MAZES_GPUUTILS_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>

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

/**
 * @brief Adaptive execution configuration parameters for GPU maze solvers
 */
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

/**
 * @brief Common OpenCL Context, Device and Program Manager
 *
 * Encapsulates device discovery, context & command queue initialization,
 * kernel source resolution, and OpenCL program compilation.
 */
struct GpuContext {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    std::string platform_name;
    std::string device_name;
    bool ready = false;

    /**
     * @brief Disables driver caching and initializes OpenCL device, context, and queue.
     */
    bool init(const std::string &preferred_vendor = "any", const std::string &solver_tag = "GPU") {
        if (ready) return true;

        setenv("LOOPY_NO_CACHE", "1", 1);
        setenv("PYOPENCL_NO_CACHE", "1", 1);
        setenv("POCL_KERNEL_CACHE", "0", 1);
        setenv("CUDA_CACHE_DISABLE", "1", 1);

        try {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) {
                std::cerr << solver_tag << " Error: No OpenCL platforms found.\n";
                return false;
            }

            bool found = false;
            std::string preferred_lower = preferred_vendor;
            std::transform(preferred_lower.begin(), preferred_lower.end(), preferred_lower.begin(), ::tolower);

            for (const auto &p : platforms) {
                std::vector<cl::Device> devices;
                p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (const auto &d : devices) {
                    std::string dev_vendor = d.getInfo<CL_DEVICE_VENDOR>();
                    std::string dev_name = d.getInfo<CL_DEVICE_NAME>();
                    std::string plat_name = p.getInfo<CL_PLATFORM_NAME>();

                    std::string dev_vendor_lower = dev_vendor;
                    std::transform(dev_vendor_lower.begin(), dev_vendor_lower.end(), dev_vendor_lower.begin(), ::tolower);

                    if (preferred_lower == "any" || dev_vendor_lower.find(preferred_lower) != std::string::npos) {
                        platform = p;
                        device = d;
                        platform_name = plat_name;
                        device_name = dev_name;
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            if (!found) {
                for (const auto &p : platforms) {
                    std::vector<cl::Device> devices;
                    p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                    if (!devices.empty()) {
                        platform = p;
                        device = devices.front();
                        platform_name = p.getInfo<CL_PLATFORM_NAME>();
                        device_name = device.getInfo<CL_DEVICE_NAME>();
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                std::cerr << solver_tag << " Error: No suitable OpenCL device found.\n";
                return false;
            }

            context = cl::Context(device);
            queue = cl::CommandQueue(context, device);
            ready = true;

            std::cout << solver_tag << " OpenCL Initialized on "
                      << device_name << " (" << platform_name << ")\n";
            return true;

        } catch (const cl::Error &err) {
            std::cerr << solver_tag << " OpenCL Exception: " << err.what()
                      << " (" << err.err() << ")\n";
            ready = false;
            return false;
        }
    }

    /**
     * @brief Loads kernel source from filesystem or falls back to embedded string
     */
    static std::string loadKernelSource(const std::string &kernel_filename, const char* embedded_source) {
        std::vector<std::filesystem::path> search_paths;
#ifdef PROJECT_ROOT_DIR
        search_paths.push_back(std::filesystem::path(PROJECT_ROOT_DIR) / "src" / "solvers" / "gpu_kernels" / kernel_filename);
#endif
        search_paths.push_back(std::filesystem::current_path() / "src" / "solvers" / "gpu_kernels" / kernel_filename);
        search_paths.push_back(std::filesystem::current_path() / "gpu_kernels" / kernel_filename);

        for (const auto &p : search_paths) {
            if (std::filesystem::exists(p)) {
                std::ifstream kf(p);
                if (kf) {
                    return std::string((std::istreambuf_iterator<char>(kf)), std::istreambuf_iterator<char>());
                }
            }
        }

        return embedded_source ? std::string(embedded_source) : std::string();
    }

    /**
     * @brief Builds an OpenCL program with compiler log reporting
     */
    cl::Program buildProgram(
        const std::string &source,
        const std::string &build_options = "-cl-std=CL2.0 -cl-fast-relaxed-math",
        const std::string &solver_tag = "GPU"
    ) {
        if (!ready) return cl::Program();

        try {
            cl::Program::Sources sources = {{source.c_str(), source.length()}};
            cl::Program prog(context, sources);

            if (prog.build({device}, build_options.c_str()) != CL_SUCCESS) {
                std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << solver_tag << " Kernel compilation failed:\n" << log << std::endl;
                ready = false;
                return cl::Program();
            }

            return prog;
        } catch (const cl::Error &err) {
            std::string log;
            try {
                cl::Program::Sources sources = {{source.c_str(), source.length()}};
                cl::Program prog(context, sources);
                prog.build({device}, build_options.c_str());
            } catch (...) {
                // Ignore secondary error while attempting to read log
            }
            std::cerr << solver_tag << " Kernel Build Exception: " << err.what()
                      << " (" << err.err() << ")\n";
            ready = false;
            return cl::Program();
        }
    }
};

#endif // MAZES_GPUUTILS_HPP
