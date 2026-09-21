#include "GpuBidirectionalGbfsSolver.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>
#include "GpuUtils.hpp"

static const char* EMBEDDED_FRONTIER_GBFS_KERNEL = R"(
#define DIR_UP    0
#define DIR_RIGHT 1
#define DIR_DOWN  2
#define DIR_LEFT  3

#define FORWARD_TAG  0x10
#define BACKWARD_TAG 0x20

inline int manhattan_dist(int r1, int c1, int r2, int c2) {
    return abs(r1 - r2) + abs(c1 - c2);
}

__kernel void expand_gbfs_batched(
    __global int* f_pri_curr,
    __global int* f_pri_next,
    __global int* f_sec_queue,
    __global int* b_pri_curr,
    __global int* b_pri_next,
    __global int* b_sec_queue,
    __global int* f_pri_count,
    __global int* f_sec_head,
    __global int* f_sec_tail,
    __global int* b_pri_count,
    __global int* b_sec_head,
    __global int* b_sec_tail,
    __global int* state_grid,
    const int height,
    const int width,
    const int start_r,
    const int start_c,
    const int end_r,
    const int end_c,
    const int batch_limit,
    const int frontier_cap,
    __global int* collision_found,
    __global int* collision_cell_A,
    __global int* collision_cell_B,
    __global int* steps_executed,
    __global int* cells_visited_acc
) {
    __local int l_f_pri_curr_count;
    __local int l_f_pri_next_count;
    __local int l_f_sec_head;
    __local int l_f_sec_tail;

    __local int l_b_pri_curr_count;
    __local int l_b_pri_next_count;
    __local int l_b_sec_head;
    __local int l_b_sec_tail;

    __local int l_collision;
    __local int l_step;
    __local int l_f_buf0;
    __local int l_b_buf0;
    __local int l_cells_visited;

    const int tid = get_local_id(0);
    const int lsize = get_local_size(0);

    if (tid == 0) {
        l_f_pri_curr_count = *f_pri_count;
        l_f_pri_next_count = 0;
        l_f_sec_head = *f_sec_head;
        l_f_sec_tail = *f_sec_tail;

        l_b_pri_curr_count = *b_pri_count;
        l_b_pri_next_count = 0;
        l_b_sec_head = *b_sec_head;
        l_b_sec_tail = *b_sec_tail;

        l_collision = *collision_found;
        l_step = 0;
        l_f_buf0 = 1;
        l_b_buf0 = 1;
        l_cells_visited = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dr[4] = {-1, 0, 1, 0};
    const int dc[4] = {0, 1, 0, -1};
    const int pdir[4] = {DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT};

    while (l_step < batch_limit && l_collision == 0 &&
           (l_f_pri_curr_count > 0 || l_f_sec_head < l_f_sec_tail) &&
           (l_b_pri_curr_count > 0 || l_b_sec_head < l_b_sec_tail)) {

        // --- Refill primary from secondary if empty ---
        if (l_f_pri_curr_count == 0 && l_f_sec_head < l_f_sec_tail) {
            const int available = l_f_sec_tail - l_f_sec_head;
            const int to_take = min(available, lsize);
            for (int i = tid; i < to_take; i += lsize) {
                __global int* pri_buf = l_f_buf0 ? f_pri_curr : f_pri_next;
                const int q_idx = (l_f_sec_head + i) % frontier_cap;
                pri_buf[i] = f_sec_queue[q_idx];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid == 0) {
                l_f_sec_head += to_take;
                l_f_pri_curr_count = to_take;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (l_b_pri_curr_count == 0 && l_b_sec_head < l_b_sec_tail) {
            const int available = l_b_sec_tail - l_b_sec_head;
            const int to_take = min(available, lsize);
            for (int i = tid; i < to_take; i += lsize) {
                __global int* pri_buf = l_b_buf0 ? b_pri_curr : b_pri_next;
                const int q_idx = (l_b_sec_head + i) % frontier_cap;
                pri_buf[i] = b_sec_queue[q_idx];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid == 0) {
                l_b_sec_head += to_take;
                l_b_pri_curr_count = to_take;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // --- 1. Forward Expansion (Target is End) ---
        {
            __global int* in_buf = l_f_buf0 ? f_pri_curr : f_pri_next;
            __global int* out_buf = l_f_buf0 ? f_pri_next : f_pri_curr;

            for (int i = tid; i < l_f_pri_curr_count; i += lsize) {
                if (l_collision != 0) break;
                const int curr_idx = in_buf[i];
                const int r = curr_idx / width;
                const int c = curr_idx % width;
                const int curr_dist = manhattan_dist(r, c, end_r, end_c);

                for (int d = 0; d < 4; ++d) {
                    const int nr = r + dr[d];
                    const int nc = c + dc[d];
                    if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
                    const int n_idx = nr * width + nc;
                    const int n_val = state_grid[n_idx];
                    if (n_val == 0xFF) continue;

                    if ((n_val & 0xF0) == BACKWARD_TAG) {
                        *collision_cell_A = curr_idx;
                        *collision_cell_B = n_idx;
                        *collision_found = 1;
                        l_collision = 1;
                        break;
                    }
                    if (n_val == 0) {
                        const int old = atomic_cmpxchg(&state_grid[n_idx], 0, FORWARD_TAG | pdir[d]);
                        if (old == 0) {
                            const int n_dist = manhattan_dist(nr, nc, end_r, end_c);
                            if (n_dist <= curr_dist) {
                                const int pos = atomic_inc(&l_f_pri_next_count);
                                if (pos < frontier_cap) {
                                    out_buf[pos] = n_idx;
                                }
                            } else {
                                const int pos = atomic_inc(&l_f_sec_tail);
                                f_sec_queue[pos % frontier_cap] = n_idx;
                            }
                            atomic_inc(&l_cells_visited);
                        } else if ((old & 0xF0) == BACKWARD_TAG) {
                            *collision_cell_A = curr_idx;
                            *collision_cell_B = n_idx;
                            *collision_found = 1;
                            l_collision = 1;
                            break;
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_collision != 0) break;

        // --- 2. Backward Expansion (Target is Start) ---
        {
            __global int* in_buf = l_b_buf0 ? b_pri_curr : b_pri_next;
            __global int* out_buf = l_b_buf0 ? b_pri_next : b_pri_curr;

            for (int i = tid; i < l_b_pri_curr_count; i += lsize) {
                if (l_collision != 0) break;
                const int curr_idx = in_buf[i];
                const int r = curr_idx / width;
                const int c = curr_idx % width;
                const int curr_dist = manhattan_dist(r, c, start_r, start_c);

                for (int d = 0; d < 4; ++d) {
                    const int nr = r + dr[d];
                    const int nc = c + dc[d];
                    if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
                    const int n_idx = nr * width + nc;
                    const int n_val = state_grid[n_idx];
                    if (n_val == 0xFF) continue;

                    if ((n_val & 0xF0) == FORWARD_TAG) {
                        *collision_cell_A = n_idx;
                        *collision_cell_B = curr_idx;
                        *collision_found = 1;
                        l_collision = 1;
                        break;
                    }
                    if (n_val == 0) {
                        const int old = atomic_cmpxchg(&state_grid[n_idx], 0, BACKWARD_TAG | pdir[d]);
                        if (old == 0) {
                            const int n_dist = manhattan_dist(nr, nc, start_r, start_c);
                            if (n_dist <= curr_dist) {
                                const int pos = atomic_inc(&l_b_pri_next_count);
                                if (pos < frontier_cap) {
                                    out_buf[pos] = n_idx;
                                }
                            } else {
                                const int pos = atomic_inc(&l_b_sec_tail);
                                b_sec_queue[pos % frontier_cap] = n_idx;
                            }
                            atomic_inc(&l_cells_visited);
                        } else if ((old & 0xF0) == FORWARD_TAG) {
                            *collision_cell_A = n_idx;
                            *collision_cell_B = curr_idx;
                            *collision_found = 1;
                            l_collision = 1;
                            break;
                        }
                    }
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_collision != 0) break;

        // --- 3. Advance to next step ---
        if (tid == 0) {
            l_step++;
            l_f_pri_curr_count = min(l_f_pri_next_count, frontier_cap);
            l_f_pri_next_count = 0;
            l_b_pri_curr_count = min(l_b_pri_next_count, frontier_cap);
            l_b_pri_next_count = 0;
            l_f_buf0 = !l_f_buf0;
            l_b_buf0 = !l_b_buf0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Ensure curr buffers contain active state on exit
    if (l_f_buf0 == 0) {
        for (int i = tid; i < l_f_pri_curr_count; i += lsize) f_pri_curr[i] = f_pri_next[i];
    }
    if (l_b_buf0 == 0) {
        for (int i = tid; i < l_b_pri_curr_count; i += lsize) b_pri_curr[i] = b_pri_next[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        *f_pri_count = l_f_pri_curr_count;
        *f_sec_head = l_f_sec_head;
        *f_sec_tail = l_f_sec_tail;
        *b_pri_count = l_b_pri_curr_count;
        *b_sec_head = l_b_sec_head;
        *b_sec_tail = l_b_sec_tail;
        atomic_add(steps_executed, l_step);
        atomic_add(cells_visited_acc, l_cells_visited);
    }
}
)";

struct GpuBidirectionalGbfsSolver::Impl {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel_gbfs_batched;
    bool ready = false;
};

static std::filesystem::path get_kernel_path() {
#ifdef PROJECT_ROOT_DIR
    return std::filesystem::path(PROJECT_ROOT_DIR) / "src" / "solvers" / "gpu_kernels" / "frontier_gbfs.cl";
#else
    return std::filesystem::current_path() / "src" / "solvers" / "gpu_kernels" / "frontier_gbfs.cl";
#endif
}

GpuBidirectionalGbfsSolver::GpuBidirectionalGbfsSolver(std::string device_vendor)
    : preferred_device_vendor(std::move(device_vendor)),
      pimpl(std::make_unique<Impl>()) {
    this->solver_name = "gpu-bidir-gbfs";
}

GpuBidirectionalGbfsSolver::~GpuBidirectionalGbfsSolver() = default;
GpuBidirectionalGbfsSolver::GpuBidirectionalGbfsSolver(GpuBidirectionalGbfsSolver&&) noexcept = default;
GpuBidirectionalGbfsSolver& GpuBidirectionalGbfsSolver::operator=(GpuBidirectionalGbfsSolver&&) noexcept = default;

bool GpuBidirectionalGbfsSolver::ensureOpenCLInitialized() {
    if (pimpl->ready) return true;

    setenv("LOOPY_NO_CACHE", "1", 1);
    setenv("PYOPENCL_NO_CACHE", "1", 1);
    setenv("POCL_KERNEL_CACHE", "0", 1);
    setenv("CUDA_CACHE_DISABLE", "1", 1);

    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "[GpuBidirectionalGbfsSolver] Error: No OpenCL platforms found.\n";
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
            std::cerr << "[GpuBidirectionalGbfsSolver] Error: No suitable OpenCL device found.\n";
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
            kernel_source = EMBEDDED_FRONTIER_GBFS_KERNEL;
        }

        cl::Program::Sources sources = {{kernel_source.c_str(), kernel_source.length()}};
        pimpl->program = cl::Program(pimpl->context, sources);

        if (pimpl->program.build({pimpl->device}, "-cl-std=CL2.0 -cl-fast-relaxed-math") != CL_SUCCESS) {
            std::string log = pimpl->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(pimpl->device);
            std::cerr << "[GpuBidirectionalGbfsSolver] Kernel compilation failed:\n" << log << std::endl;
            return false;
        }

        pimpl->kernel_gbfs_batched = cl::Kernel(pimpl->program, "expand_gbfs_batched");
        pimpl->ready = true;

        std::cout << "[GpuBidirectionalGbfsSolver] OpenCL Initialized on "
                  << selected_device_name << " (" << selected_platform_name << ")\n";
        return true;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuBidirectionalGbfsSolver] OpenCL Exception: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    }
}

bool GpuBidirectionalGbfsSolver::solve(const Maze &maze_object) {
    if (!ensureOpenCLInitialized()) {
        std::cerr << "[GpuBidirectionalGbfsSolver] Failed to initialize OpenCL runtime.\n";
        return false;
    }

    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();
    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

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
        cl::Buffer buf_state(pimpl->context, CL_MEM_READ_WRITE, total_cells * sizeof(int));
        pimpl->queue.enqueueWriteBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

        // Compute adaptive GPU execution parameters
        const auto adaptive_cfg = computeGpuAdaptiveConfig(
            pimpl->device,
            this->height,
            this->width,
            sizeof(int),
            this->user_batch_size
        );
        this->config_rationale = adaptive_cfg.rationale;
        const int BATCH_SIZE = adaptive_cfg.batch_size;
        const size_t LOCAL_WORKGROUP_SIZE = adaptive_cfg.workgroup_size;
        const size_t FRONTIER_CAP = adaptive_cfg.frontier_capacity;

        cl::Buffer buf_f_pri_curr(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_f_pri_next(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_f_sec_queue(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));

        cl::Buffer buf_b_pri_curr(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_b_pri_next(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_b_sec_queue(pimpl->context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));

        cl::Buffer buf_f_pri_count(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_f_sec_head(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_f_sec_tail(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

        cl::Buffer buf_b_pri_count(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_b_sec_head(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_b_sec_tail(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

        cl::Buffer buf_collision_found(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_A(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_B(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_steps_executed(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_cells_visited(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

        const int one = 1;
        const int zero = 0;
        pimpl->queue.enqueueWriteBuffer(buf_f_pri_curr, CL_FALSE, 0, sizeof(int), &start_idx);
        pimpl->queue.enqueueWriteBuffer(buf_b_pri_curr, CL_FALSE, 0, sizeof(int), &end_idx);

        pimpl->queue.enqueueWriteBuffer(buf_f_pri_count, CL_FALSE, 0, sizeof(int), &one);
        pimpl->queue.enqueueWriteBuffer(buf_f_sec_head, CL_FALSE, 0, sizeof(int), &zero);
        pimpl->queue.enqueueWriteBuffer(buf_f_sec_tail, CL_FALSE, 0, sizeof(int), &zero);

        pimpl->queue.enqueueWriteBuffer(buf_b_pri_count, CL_FALSE, 0, sizeof(int), &one);
        pimpl->queue.enqueueWriteBuffer(buf_b_sec_head, CL_FALSE, 0, sizeof(int), &zero);
        pimpl->queue.enqueueWriteBuffer(buf_b_sec_tail, CL_FALSE, 0, sizeof(int), &zero);

        pimpl->queue.enqueueWriteBuffer(buf_collision_found, CL_FALSE, 0, sizeof(int), &zero);
        pimpl->queue.enqueueWriteBuffer(buf_steps_executed, CL_FALSE, 0, sizeof(int), &zero);
        pimpl->queue.enqueueWriteBuffer(buf_cells_visited, CL_TRUE, 0, sizeof(int), &zero);

        int f_pri_c = 1, f_head = 0, f_tail = 0;
        int b_pri_c = 1, b_head = 0, b_tail = 0;
        total_frontier_expansions = 0;
        cells_visited = 2;
        int collision_found = 0;
        int cell_A = -1;
        int cell_B = -1;

        pimpl->kernel_gbfs_batched.setArg(0, buf_f_pri_curr);
        pimpl->kernel_gbfs_batched.setArg(1, buf_f_pri_next);
        pimpl->kernel_gbfs_batched.setArg(2, buf_f_sec_queue);
        pimpl->kernel_gbfs_batched.setArg(3, buf_b_pri_curr);
        pimpl->kernel_gbfs_batched.setArg(4, buf_b_pri_next);
        pimpl->kernel_gbfs_batched.setArg(5, buf_b_sec_queue);
        pimpl->kernel_gbfs_batched.setArg(6, buf_f_pri_count);
        pimpl->kernel_gbfs_batched.setArg(7, buf_f_sec_head);
        pimpl->kernel_gbfs_batched.setArg(8, buf_f_sec_tail);
        pimpl->kernel_gbfs_batched.setArg(9, buf_b_pri_count);
        pimpl->kernel_gbfs_batched.setArg(10, buf_b_sec_head);
        pimpl->kernel_gbfs_batched.setArg(11, buf_b_sec_tail);
        pimpl->kernel_gbfs_batched.setArg(12, buf_state);
        pimpl->kernel_gbfs_batched.setArg(13, this->height);
        pimpl->kernel_gbfs_batched.setArg(14, this->width);
        pimpl->kernel_gbfs_batched.setArg(15, this->start.first);
        pimpl->kernel_gbfs_batched.setArg(16, this->start.second);
        pimpl->kernel_gbfs_batched.setArg(17, this->end.first);
        pimpl->kernel_gbfs_batched.setArg(18, this->end.second);
        pimpl->kernel_gbfs_batched.setArg(19, BATCH_SIZE);
        pimpl->kernel_gbfs_batched.setArg(20, static_cast<int>(FRONTIER_CAP));
        pimpl->kernel_gbfs_batched.setArg(21, buf_collision_found);
        pimpl->kernel_gbfs_batched.setArg(22, buf_collision_cell_A);
        pimpl->kernel_gbfs_batched.setArg(23, buf_collision_cell_B);
        pimpl->kernel_gbfs_batched.setArg(24, buf_steps_executed);
        pimpl->kernel_gbfs_batched.setArg(25, buf_cells_visited);

        while (collision_found == 0 && (f_pri_c > 0 || f_head < f_tail) && (b_pri_c > 0 || b_head < b_tail)) {
            pimpl->queue.enqueueNDRangeKernel(
                pimpl->kernel_gbfs_batched,
                cl::NullRange,
                cl::NDRange(LOCAL_WORKGROUP_SIZE),
                cl::NDRange(LOCAL_WORKGROUP_SIZE)
            );

            pimpl->queue.enqueueReadBuffer(buf_collision_found, CL_FALSE, 0, sizeof(int), &collision_found);
            pimpl->queue.enqueueReadBuffer(buf_f_pri_count, CL_FALSE, 0, sizeof(int), &f_pri_c);
            pimpl->queue.enqueueReadBuffer(buf_f_sec_head, CL_FALSE, 0, sizeof(int), &f_head);
            pimpl->queue.enqueueReadBuffer(buf_f_sec_tail, CL_FALSE, 0, sizeof(int), &f_tail);
            pimpl->queue.enqueueReadBuffer(buf_b_pri_count, CL_FALSE, 0, sizeof(int), &b_pri_c);
            pimpl->queue.enqueueReadBuffer(buf_b_sec_head, CL_FALSE, 0, sizeof(int), &b_head);
            pimpl->queue.enqueueReadBuffer(buf_b_sec_tail, CL_TRUE, 0, sizeof(int), &b_tail);

            if (collision_found != 0) break;
        }

        int steps_executed = 0;
        int visited_count = 0;
        pimpl->queue.enqueueReadBuffer(buf_steps_executed, CL_FALSE, 0, sizeof(int), &steps_executed);
        pimpl->queue.enqueueReadBuffer(buf_cells_visited, CL_TRUE, 0, sizeof(int), &visited_count);
        total_frontier_expansions = steps_executed;
        cells_visited = visited_count + 2;

        if (collision_found == 0) {
            std::cerr << "[GpuBidirectionalGbfsSolver] No path found between Start and End.\n";
            return false;
        }

        pimpl->queue.enqueueReadBuffer(buf_collision_cell_A, CL_FALSE, 0, sizeof(int), &cell_A);
        pimpl->queue.enqueueReadBuffer(buf_collision_cell_B, CL_FALSE, 0, sizeof(int), &cell_B);
        pimpl->queue.enqueueReadBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

        // Path Reconstruction
        this->solution.assign(this->height, std::vector<bool>(this->width, false));
        this->visited.assign(this->height, std::vector<bool>(this->width, false));

        for (int r = 0; r < this->height; ++r) {
            const size_t row_offset = static_cast<size_t>(r) * this->width;
            for (int c = 0; c < this->width; ++c) {
                const int val = host_state[row_offset + c];
                if (val != 0 && val != 0xFF) {
                    this->visited[r][c] = true;
                }
            }
        }

        size_t path_len = 0;
        const int dr[4] = {-1, 0, 1, 0};
        const int dc[4] = {0, 1, 0, -1};

        // Forward traceback: cell_A -> start
        int curr = cell_A;
        while (curr >= 0) {
            const int r = curr / this->width;
            const int c = curr % this->width;
            this->solution[r][c] = true;
            path_len++;

            if (r == this->start.first && c == this->start.second) break;

            const int val = host_state[curr];
            const int pdir = val & 0x03;
            const int pr = r + dr[pdir];
            const int pc = c + dc[pdir];
            if (pr < 0 || pr >= this->height || pc < 0 || pc >= this->width) break;
            curr = pr * this->width + pc;
        }

        // Backward traceback: cell_B -> end
        curr = cell_B;
        while (curr >= 0) {
            const int r = curr / this->width;
            const int c = curr % this->width;
            if (!this->solution[r][c]) {
                this->solution[r][c] = true;
                path_len++;
            }

            if (r == this->end.first && c == this->end.second) break;

            const int val = host_state[curr];
            const int pdir = val & 0x03;
            const int pr = r + dr[pdir];
            const int pc = c + dc[pdir];
            if (pr < 0 || pr >= this->height || pc < 0 || pc >= this->width) break;
            curr = pr * this->width + pc;
        }

        std::cout << "[GpuBidirectionalGbfsSolver] Path found in " << total_frontier_expansions
                  << " frontier steps (" << cells_visited << " cells visited)\n";

        return (path_len > 0 &&
                this->solution[this->start.first][this->start.second] &&
                this->solution[this->end.first][this->end.second]);

    } catch (const cl::Error &err) {
        std::cerr << "[GpuBidirectionalGbfsSolver] OpenCL error: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    }
}
