#include "GpuDeadEndFillingSolver.hpp"

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

// Embedded fallback OpenCL kernel string in case external file is not present
static const char* EMBEDDED_KERNEL_SOURCE = R"(
__kernel void dead_end_filling_step(
    __global const uchar* in_grid,
    __global uchar* out_grid,
    __global int* change_count,
    const int height,
    const int width,
    const int start_r,
    const int start_c,
    const int end_r,
    const int end_c
) {
    const int c = get_global_id(0);
    const int r = get_global_id(1);

    __local int l_changes;
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        l_changes = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    bool pruned = false;

    if (r < height && c < width) {
        const int idx = r * width + c;
        const uchar cell = in_grid[idx];

        if (cell != 0) {
            out_grid[idx] = 1;
        } else if ((r == start_r && c == start_c) || (r == end_r && c == end_c)) {
            out_grid[idx] = 0;
        } else {
            int open_count = 0;
            if (r > 0 && in_grid[(r - 1) * width + c] == 0) open_count++;
            if (r < height - 1 && in_grid[(r + 1) * width + c] == 0) open_count++;
            if (c > 0 && in_grid[r * width + (c - 1)] == 0) open_count++;
            if (c < width - 1 && in_grid[r * width + (c + 1)] == 0) open_count++;

            if (open_count <= 1) {
                out_grid[idx] = 1;
                pruned = true;
            } else {
                out_grid[idx] = 0;
            }
        }
    }

    if (pruned) {
        atomic_inc(&l_changes);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        if (l_changes > 0) {
            atomic_add(change_count, l_changes);
        }
    }
}
)";

struct GpuDeadEndFillingSolver::Impl {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    bool ready = false;
};

static std::filesystem::path get_kernel_path() {
#ifdef PROJECT_ROOT_DIR
    return std::filesystem::path(PROJECT_ROOT_DIR) / "src" / "solvers" / "gpu_kernels" / "dead_end_filling.cl";
#else
    return std::filesystem::current_path() / "src" / "solvers" / "gpu_kernels" / "dead_end_filling.cl";
#endif
}

GpuDeadEndFillingSolver::GpuDeadEndFillingSolver(std::string device_vendor)
    : preferred_device_vendor(std::move(device_vendor)),
      pimpl(std::make_unique<Impl>()) {
    this->solver_name = "gpu-dead-end";
}

GpuDeadEndFillingSolver::~GpuDeadEndFillingSolver() = default;

GpuDeadEndFillingSolver::GpuDeadEndFillingSolver(GpuDeadEndFillingSolver&&) noexcept = default;
GpuDeadEndFillingSolver& GpuDeadEndFillingSolver::operator=(GpuDeadEndFillingSolver&&) noexcept = default;

bool GpuDeadEndFillingSolver::isGpuAvailable() {
    return ensureOpenCLInitialized();
}

bool GpuDeadEndFillingSolver::ensureOpenCLInitialized() {
    if (pimpl->ready) {
        return true;
    }

    // Critical cache prevention environment variables (matches seizure-algorithm reference)
    setenv("LOOPY_NO_CACHE", "1", 1);
    setenv("PYOPENCL_NO_CACHE", "1", 1);
    setenv("POCL_KERNEL_CACHE", "0", 1);
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "[GpuDeadEndFillingSolver] Error: No OpenCL platforms found.\n";
            return false;
        }

        bool found = false;
        std::string preferred_lower = preferred_device_vendor;
        std::transform(preferred_lower.begin(), preferred_lower.end(), preferred_lower.begin(), ::tolower);

        // Pass 1: Search for preferred GPU or any GPU
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
                    pimpl->platform = p;
                    pimpl->device = d;
                    selected_platform_name = plat_name;
                    selected_device_name = dev_name;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }

        // Pass 2: Fallback to any available device (CPU / Accelerator / POCL)
        if (!found) {
            for (const auto &p : platforms) {
                std::vector<cl::Device> devices;
                p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                if (!devices.empty()) {
                    pimpl->platform = p;
                    pimpl->device = devices.front();
                    selected_platform_name = p.getInfo<CL_PLATFORM_NAME>();
                    selected_device_name = pimpl->device.getInfo<CL_DEVICE_NAME>();
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            std::cerr << "[GpuDeadEndFillingSolver] Error: No suitable OpenCL device found.\n";
            return false;
        }

        pimpl->context = cl::Context(pimpl->device);
        pimpl->queue = cl::CommandQueue(pimpl->context, pimpl->device);

        // Load kernel source (prefer external .cl file if available, otherwise embedded fallback)
        std::string kernel_source;
        const auto kernel_path = get_kernel_path();
        std::ifstream kf(kernel_path);
        if (kf) {
            kernel_source.assign((std::istreambuf_iterator<char>(kf)), std::istreambuf_iterator<char>());
        } else {
            kernel_source = EMBEDDED_KERNEL_SOURCE;
        }

        cl::Program::Sources sources = {{kernel_source.c_str(), kernel_source.length()}};
        pimpl->program = cl::Program(pimpl->context, sources);

        if (pimpl->program.build({pimpl->device}, "-cl-std=CL2.0 -cl-fast-relaxed-math") != CL_SUCCESS) {
            std::string log = pimpl->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(pimpl->device);
            std::cerr << "[GpuDeadEndFillingSolver] Kernel compilation failed:\n" << log << std::endl;
            return false;
        }

        pimpl->kernel = cl::Kernel(pimpl->program, "dead_end_filling_step");
        pimpl->ready = true;
        is_initialized = true;

        std::cout << "[GpuDeadEndFillingSolver] OpenCL Initialized on "
                  << selected_device_name << " (" << selected_platform_name << ")\n";
        return true;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuDeadEndFillingSolver] OpenCL Exception: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    } catch (const std::exception &ex) {
        std::cerr << "[GpuDeadEndFillingSolver] Exception: " << ex.what() << "\n";
        return false;
    }
}

bool GpuDeadEndFillingSolver::solve(const Maze &maze_object) {
    if (!ensureOpenCLInitialized()) {
        std::cerr << "[GpuDeadEndFillingSolver] Failed to initialize OpenCL runtime.\n";
        return false;
    }

    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();
    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

    // Check device VRAM availability
    const cl_ulong global_mem_bytes = pimpl->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    const size_t required_bytes = total_cells * 2 + sizeof(int) * 4;
    if (required_bytes > global_mem_bytes) {
        std::cerr << "[GpuDeadEndFillingSolver] Error: Maze requires "
                  << (required_bytes / (1024 * 1024)) << " MB VRAM, but device has only "
                  << (global_mem_bytes / (1024 * 1024)) << " MB.\n";
        return false;
    }

    // Flatten host maze grid: 0 for PATH, 1 for WALL
    std::vector<uint8_t> host_grid(total_cells);
    for (int r = 0; r < this->height; ++r) {
        const size_t row_offset = static_cast<size_t>(r) * this->width;
        for (int c = 0; c < this->width; ++c) {
            host_grid[row_offset + c] = grid[r][c] ? 1 : 0;
        }
    }

    try {
        // Allocate ping-pong buffers and atomic change counter in VRAM
        cl::Buffer buf_in(pimpl->context, CL_MEM_READ_WRITE, total_cells * sizeof(uint8_t));
        cl::Buffer buf_out(pimpl->context, CL_MEM_READ_WRITE, total_cells * sizeof(uint8_t));
        cl::Buffer buf_changes(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

        // Upload initial maze to buf_in
        pimpl->queue.enqueueWriteBuffer(buf_in, CL_TRUE, 0, total_cells * sizeof(uint8_t), host_grid.data());

        // 2D tile work size (16x16 = 256 work-items per work-group)
        constexpr size_t TILE_SIZE = 16;
        const cl::NDRange local_range(TILE_SIZE, TILE_SIZE);
        const cl::NDRange global_range(
            ((this->width + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE,
            ((this->height + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE
        );

        total_iterations = 0;
        int changes = 0;
        const int zero = 0;

        // Cellular Automaton Iteration Loop
        while (true) {
            // Reset change counter to zero
            pimpl->queue.enqueueWriteBuffer(buf_changes, CL_FALSE, 0, sizeof(int), &zero);

            // Set kernel arguments
            pimpl->kernel.setArg(0, buf_in);
            pimpl->kernel.setArg(1, buf_out);
            pimpl->kernel.setArg(2, buf_changes);
            pimpl->kernel.setArg(3, this->height);
            pimpl->kernel.setArg(4, this->width);
            pimpl->kernel.setArg(5, this->start.first);
            pimpl->kernel.setArg(6, this->start.second);
            pimpl->kernel.setArg(7, this->end.first);
            pimpl->kernel.setArg(8, this->end.second);

            // Launch parallel kernel step
            pimpl->queue.enqueueNDRangeKernel(pimpl->kernel, cl::NullRange, global_range, local_range);

            // Read back change counter (blocking read ensures GPU kernel completion)
            pimpl->queue.enqueueReadBuffer(buf_changes, CL_TRUE, 0, sizeof(int), &changes);

            total_iterations++;
            std::swap(buf_in, buf_out);

            // If no cells were pruned in this step, convergence is reached!
            if (changes == 0) {
                break;
            }
        }

        // Read back final converged grid from buf_in (which holds the output of the last step)
        pimpl->queue.enqueueReadBuffer(buf_in, CL_TRUE, 0, total_cells * sizeof(uint8_t), host_grid.data());

        // Populate solution and visited maps
        this->solution.assign(this->height, std::vector<bool>(this->width, false));
        this->visited.assign(this->height, std::vector<bool>(this->width, false));

        size_t solution_cells = 0;
        for (int r = 0; r < this->height; ++r) {
            const size_t row_offset = static_cast<size_t>(r) * this->width;
            for (int c = 0; c < this->width; ++c) {
                const bool original_is_path = !grid[r][c];
                const bool surviving_is_path = (host_grid[row_offset + c] == 0);

                if (surviving_is_path) {
                    this->solution[r][c] = true;
                    this->visited[r][c] = true;
                    solution_cells++;
                } else if (original_is_path) {
                    // Pruned dead end
                    this->visited[r][c] = true;
                }
            }
        }

        const bool solvable = (solution_cells > 0 &&
                               this->solution[this->start.first][this->start.second] &&
                               this->solution[this->end.first][this->end.second]);

        std::cout << "[GpuDeadEndFillingSolver] Converged in " << total_iterations
                  << " parallel GPU steps (" << solution_cells << " solution cells)\n";

        return solvable;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuDeadEndFillingSolver] OpenCL runtime error: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    }
}
