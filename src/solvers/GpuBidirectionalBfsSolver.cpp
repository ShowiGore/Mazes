#include "GpuBidirectionalBfsSolver.hpp"

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

static const char* EMBEDDED_FRONTIER_BFS_KERNEL = R"(
#define DIR_UP    0
#define DIR_RIGHT 1
#define DIR_DOWN  2
#define DIR_LEFT  3

#define FORWARD_TAG  0x10
#define BACKWARD_TAG 0x20

__kernel void expand_frontier_step(
    __global const int* curr_frontier,
    const int curr_count,
    __global int* next_frontier,
    __global int* next_count,
    __global int* state_grid,
    const int height,
    const int width,
    const int search_tag,
    const int opp_tag,
    __global int* collision_found,
    __global int* collision_cell_A,
    __global int* collision_cell_B
) {
    const int tid = get_global_id(0);
    if (tid >= curr_count || *collision_found != 0) {
        return;
    }

    const int curr_idx = curr_frontier[tid];
    const int r = curr_idx / width;
    const int c = curr_idx % width;

    const int dr[4] = {-1, 0, 1, 0};
    const int dc[4] = {0, 1, 0, -1};
    const int parent_dir[4] = {DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT};

    for (int i = 0; i < 4; ++i) {
        const int nr = r + dr[i];
        const int nc = c + dc[i];

        if (nr < 0 || nr >= height || nc < 0 || nc >= width) {
            continue;
        }

        const int n_idx = nr * width + nc;
        const int n_val = state_grid[n_idx];

        if (n_val == 0xFF) {
            continue;
        }

        if ((n_val & 0xF0) == opp_tag) {
            *collision_cell_A = curr_idx;
            *collision_cell_B = n_idx;
            atomic_xchg(collision_found, 1);
            return;
        }

        if (n_val == 0) {
            const int new_val = search_tag | parent_dir[i];
            const int old = atomic_cmpxchg(&state_grid[n_idx], 0, new_val);

            if (old == 0) {
                const int pos = atomic_inc(next_count);
                next_frontier[pos] = n_idx;
            } else if ((old & 0xF0) == opp_tag) {
                *collision_cell_A = curr_idx;
                *collision_cell_B = n_idx;
                atomic_xchg(collision_found, 1);
                return;
            }
        }
    }
}
)";

struct GpuBidirectionalBfsSolver::Impl {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Kernel kernel_batched;
    bool ready = false;
};

static std::filesystem::path get_kernel_path() {
#ifdef PROJECT_ROOT_DIR
    return std::filesystem::path(PROJECT_ROOT_DIR) / "src" / "solvers" / "gpu_kernels" / "frontier_bfs.cl";
#else
    return std::filesystem::current_path() / "src" / "solvers" / "gpu_kernels" / "frontier_bfs.cl";
#endif
}

GpuBidirectionalBfsSolver::GpuBidirectionalBfsSolver(std::string device_vendor)
    : preferred_device_vendor(std::move(device_vendor)),
      pimpl(std::make_unique<Impl>()) {
    this->solver_name = "gpu-bidir-bfs";
}

GpuBidirectionalBfsSolver::~GpuBidirectionalBfsSolver() = default;

GpuBidirectionalBfsSolver::GpuBidirectionalBfsSolver(GpuBidirectionalBfsSolver&&) noexcept = default;
GpuBidirectionalBfsSolver& GpuBidirectionalBfsSolver::operator=(GpuBidirectionalBfsSolver&&) noexcept = default;

bool GpuBidirectionalBfsSolver::ensureOpenCLInitialized() {
    if (pimpl->ready) {
        return true;
    }

    setenv("LOOPY_NO_CACHE", "1", 1);
    setenv("PYOPENCL_NO_CACHE", "1", 1);
    setenv("POCL_KERNEL_CACHE", "0", 1);
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "[GpuBidirectionalBfsSolver] Error: No OpenCL platforms found.\n";
            return false;
        }

        bool found = false;
        std::string preferred_lower = preferred_device_vendor;
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
            std::cerr << "[GpuBidirectionalBfsSolver] Error: No suitable OpenCL device found.\n";
            return false;
        }

        pimpl->context = cl::Context(pimpl->device);
        pimpl->queue = cl::CommandQueue(pimpl->context, pimpl->device);

        std::string kernel_source;
        const auto kernel_path = get_kernel_path();
        std::ifstream kf(kernel_path);
        if (kf) {
            kernel_source.assign((std::istreambuf_iterator<char>(kf)), std::istreambuf_iterator<char>());
        } else {
            kernel_source = EMBEDDED_FRONTIER_BFS_KERNEL;
        }

        cl::Program::Sources sources = {{kernel_source.c_str(), kernel_source.length()}};
        pimpl->program = cl::Program(pimpl->context, sources);

        if (pimpl->program.build({pimpl->device}, "-cl-std=CL2.0 -cl-fast-relaxed-math") != CL_SUCCESS) {
            std::string log = pimpl->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(pimpl->device);
            std::cerr << "[GpuBidirectionalBfsSolver] Kernel compilation failed:\n" << log << std::endl;
            return false;
        }

        pimpl->kernel = cl::Kernel(pimpl->program, "expand_frontier_step");
        pimpl->kernel_batched = cl::Kernel(pimpl->program, "expand_frontier_batched");
        pimpl->ready = true;

        std::cout << "[GpuBidirectionalBfsSolver] OpenCL Initialized on "
                  << selected_device_name << " (" << selected_platform_name << ")\n";
        return true;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuBidirectionalBfsSolver] OpenCL Exception: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    }
}

bool GpuBidirectionalBfsSolver::solve(const Maze &maze_object) {
    if (!ensureOpenCLInitialized()) {
        std::cerr << "[GpuBidirectionalBfsSolver] Failed to initialize OpenCL runtime.\n";
        return false;
    }

    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();
    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

    // State grid: 0x00 = PATH, 0xFF = WALL, 0x10 = Forward start, 0x20 = Backward end
    std::vector<int> host_state(total_cells);
    for (int r = 0; r < this->height; ++r) {
        const size_t row_offset = static_cast<size_t>(r) * this->width;
        for (int c = 0; c < this->width; ++c) {
            host_state[row_offset + c] = grid[r][c] ? 0xFF : 0x00;
        }
    }

    const int start_idx = this->start.first * this->width + this->start.second;
    const int end_idx = this->end.first * this->width + this->end.second;

    constexpr int FORWARD_TAG = 0x10;
    constexpr int BACKWARD_TAG = 0x20;

    host_state[start_idx] = FORWARD_TAG;
    host_state[end_idx] = BACKWARD_TAG;

    try {
        // State grid in VRAM (32-bit int per cell for native atomic CAS)
        cl::Buffer buf_state(pimpl->context, CL_MEM_READ_WRITE, total_cells * sizeof(int));
        pimpl->queue.enqueueWriteBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

        // Dynamic frontier capacity
        const size_t FRONTIER_CAP = std::min<size_t>(1000000, total_cells / 4 + 1024);
        cl::Buffer buf_f_curr(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_f_next(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_b_curr(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_b_next(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));

        cl::Buffer buf_f_count(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_b_count(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_found(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_A(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_B(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_steps_executed(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

        // Upload initial frontiers and counters
        const int one = 1;
        const int zero = 0;
        pimpl->queue.enqueueWriteBuffer(buf_f_curr, CL_FALSE, 0, sizeof(int), &start_idx);
        pimpl->queue.enqueueWriteBuffer(buf_b_curr, CL_FALSE, 0, sizeof(int), &end_idx);
        pimpl->queue.enqueueWriteBuffer(buf_f_count, CL_FALSE, 0, sizeof(int), &one);
        pimpl->queue.enqueueWriteBuffer(buf_b_count, CL_FALSE, 0, sizeof(int), &one);
        pimpl->queue.enqueueWriteBuffer(buf_collision_found, CL_FALSE, 0, sizeof(int), &zero);
        pimpl->queue.enqueueWriteBuffer(buf_steps_executed, CL_TRUE, 0, sizeof(int), &zero);

        int f_count = 1;
        int b_count = 1;
        total_frontier_expansions = 0;
        cells_visited = 2;
        int collision_found = 0;
        int cell_A = -1;
        int cell_B = -1;

        constexpr int BATCH_SIZE = 128;
        constexpr size_t LOCAL_WORKGROUP_SIZE = 512;

        pimpl->kernel_batched.setArg(0, buf_f_curr);
        pimpl->kernel_batched.setArg(1, buf_f_next);
        pimpl->kernel_batched.setArg(2, buf_b_curr);
        pimpl->kernel_batched.setArg(3, buf_b_next);
        pimpl->kernel_batched.setArg(4, buf_f_count);
        pimpl->kernel_batched.setArg(5, buf_b_count);
        pimpl->kernel_batched.setArg(6, buf_state);
        pimpl->kernel_batched.setArg(7, this->height);
        pimpl->kernel_batched.setArg(8, this->width);
        pimpl->kernel_batched.setArg(9, BATCH_SIZE);
        pimpl->kernel_batched.setArg(10, buf_collision_found);
        pimpl->kernel_batched.setArg(11, buf_collision_cell_A);
        pimpl->kernel_batched.setArg(12, buf_collision_cell_B);
        pimpl->kernel_batched.setArg(13, buf_steps_executed);

        // Batched Persistent Frontier Expansion Loop
        while (collision_found == 0 && f_count > 0 && b_count > 0) {
            pimpl->queue.enqueueNDRangeKernel(
                pimpl->kernel_batched,
                cl::NullRange,
                cl::NDRange(LOCAL_WORKGROUP_SIZE),
                cl::NDRange(LOCAL_WORKGROUP_SIZE)
            );

            // Read collision status and counts once per batch
            pimpl->queue.enqueueReadBuffer(buf_collision_found, CL_FALSE, 0, sizeof(int), &collision_found);
            pimpl->queue.enqueueReadBuffer(buf_f_count, CL_FALSE, 0, sizeof(int), &f_count);
            pimpl->queue.enqueueReadBuffer(buf_b_count, CL_TRUE, 0, sizeof(int), &b_count);

            if (collision_found != 0) break;
            if (f_count == 0 || b_count == 0) break;
        }

        int steps_executed = 0;
        pimpl->queue.enqueueReadBuffer(buf_steps_executed, CL_TRUE, 0, sizeof(int), &steps_executed);
        total_frontier_expansions = steps_executed;

        if (collision_found == 0) {
            std::cerr << "[GpuBidirectionalBfsSolver] No path found between Start and End.\n";
            return false;
        }

        pimpl->queue.enqueueReadBuffer(buf_collision_cell_A, CL_FALSE, 0, sizeof(int), &cell_A);
        pimpl->queue.enqueueReadBuffer(buf_collision_cell_B, CL_FALSE, 0, sizeof(int), &cell_B);
        pimpl->queue.enqueueReadBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

        // Path Reconstruction: follow parent pointers
        this->solution.assign(this->height, std::vector<bool>(this->width, false));
        this->visited.assign(this->height, std::vector<bool>(this->width, false));

        // Mark visited cells
        for (int r = 0; r < this->height; ++r) {
            const size_t row_offset = static_cast<size_t>(r) * this->width;
            for (int c = 0; c < this->width; ++c) {
                const int val = host_state[row_offset + c];
                if (val != 0x00 && val != 0xFF) {
                    this->visited[r][c] = true;
                }
            }
        }

        const int dr[4] = {-1, 0, 1, 0}; // UP, RIGHT, DOWN, LEFT
        const int dc[4] = {0, 1, 0, -1};

        // Determine which of cell_A / cell_B belongs to forward search and which to backward search
        const int f_start = ((host_state[cell_A] & 0xF0) == FORWARD_TAG) ? cell_A : cell_B;
        const int b_start = ((host_state[cell_A] & 0xF0) == BACKWARD_TAG) ? cell_A : cell_B;

        // Trace forward branch: f_start -> start
        int curr = f_start;
        while (curr >= 0) {
            const int r = curr / this->width;
            const int c = curr % this->width;
            this->solution[r][c] = true;

            if (r == this->start.first && c == this->start.second) {
                break;
            }

            const int val = host_state[curr];
            const int dir = val & 0x03; // Parent direction
            curr = (r + dr[dir]) * this->width + (c + dc[dir]);
        }

        // Trace backward branch: b_start -> end
        curr = b_start;
        while (curr >= 0) {
            const int r = curr / this->width;
            const int c = curr % this->width;
            this->solution[r][c] = true;

            if (r == this->end.first && c == this->end.second) {
                break;
            }

            const int val = host_state[curr];
            const int dir = val & 0x03; // Parent direction
            curr = (r + dr[dir]) * this->width + (c + dc[dir]);
        }

        std::cout << "[GpuBidirectionalBfsSolver] Path found in " << total_frontier_expansions
                  << " frontier steps (" << cells_visited << " cells visited)\n";

        return true;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuBidirectionalBfsSolver] OpenCL runtime error: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    }
}
