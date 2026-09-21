/**
 * =============================================================================
 * GPU BIDIRECTIONAL BREADTH-FIRST SEARCH (BFS) SOLVER
 * =============================================================================
 *
 * 1. ALGORITHM STRATEGY:
 *    - Dual-Frontier Expansion: Simultaneously expands a Forward frontier from 'start'
 *      and a Backward frontier from 'end' across level-synchronous BFS wavefronts.
 *    - In an unweighted planar graph, BFS guarantees finding the shortest geodesic path.
 *    - Expanding bidirectionally cuts the search radius from R to R/2, reducing the
 *      number of explored cells from O(pi * R^2) to O(2 * pi * (R/2)^2) = O(0.5 * pi * R^2).
 *    - Collision is detected atomically via compare-and-exchange on the state grid
 *      (Forward tag = 0x10, Backward tag = 0x20).
 *
 * 2. HARDWARE & MAZE ADAPTIVE EXECUTION:
 *    - Uses computeGpuAdaptiveConfig() to adaptively determine:
 *      * Workgroup size (clamped to GPU max workgroup size, aligned to warp/wavefront 32).
 *      * Frontier buffer capacity (scaled to available VRAM headroom to prevent OOM).
 *      * Batch size (scaled with maze diameter to amortize kernel launch and host sync).
 * =============================================================================
 */

#include "GpuBidirectionalBfsSolver.hpp"
#include "utilities/GpuUtils.hpp"

#include <iostream>
#include <vector>
#include <algorithm>

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
    GpuContext gpu;
    cl::Program program;
    cl::Kernel kernel;
    cl::Kernel kernel_batched;
};

GpuBidirectionalBfsSolver::GpuBidirectionalBfsSolver(std::string device_vendor)
    : GpuSolver(std::move(device_vendor)),
      pimpl(std::make_unique<Impl>()) {
    this->solver_name = "gpu-bidir-bfs";
}

GpuBidirectionalBfsSolver::~GpuBidirectionalBfsSolver() = default;

GpuBidirectionalBfsSolver::GpuBidirectionalBfsSolver(GpuBidirectionalBfsSolver&&) noexcept = default;
GpuBidirectionalBfsSolver& GpuBidirectionalBfsSolver::operator=(GpuBidirectionalBfsSolver&&) noexcept = default;

bool GpuBidirectionalBfsSolver::ensureOpenCLInitialized() {
    if (pimpl->gpu.ready) {
        return true;
    }

    if (!pimpl->gpu.init(this->preferred_device_vendor, "[GpuBidirectionalBfsSolver]")) {
        return false;
    }
    this->selected_platform_name = pimpl->gpu.platform_name;
    this->selected_device_name = pimpl->gpu.device_name;

    const std::string kernel_source = GpuContext::loadKernelSource("frontier_bfs.cl", EMBEDDED_FRONTIER_BFS_KERNEL);
    pimpl->program = pimpl->gpu.buildProgram(kernel_source, "-cl-std=CL2.0 -cl-fast-relaxed-math", "[GpuBidirectionalBfsSolver]");
    if (!pimpl->gpu.ready) {
        return false;
    }

    pimpl->kernel = cl::Kernel(pimpl->program, "expand_frontier_step");
    pimpl->kernel_batched = cl::Kernel(pimpl->program, "expand_frontier_batched");
    return true;
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
        cl::Buffer buf_state(pimpl->gpu.context, CL_MEM_READ_WRITE, total_cells * sizeof(int));
        pimpl->gpu.queue.enqueueWriteBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

        // Compute adaptive GPU execution parameters (batch size, workgroup, frontier capacity)
        const auto adaptive_cfg = computeGpuAdaptiveConfig(
            pimpl->gpu.device,
            this->height,
            this->width,
            sizeof(int),
            this->user_batch_size
        );
        this->config_rationale = adaptive_cfg.rationale;
        const int BATCH_SIZE = adaptive_cfg.batch_size;
        const size_t LOCAL_WORKGROUP_SIZE = adaptive_cfg.workgroup_size;
        const size_t FRONTIER_CAP = adaptive_cfg.frontier_capacity;

        // Dynamic frontier capacity
        cl::Buffer buf_f_curr(pimpl->gpu.context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_f_next(pimpl->gpu.context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_b_curr(pimpl->gpu.context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));
        cl::Buffer buf_b_next(pimpl->gpu.context, CL_MEM_READ_WRITE, FRONTIER_CAP * sizeof(int));

        cl::Buffer buf_f_count(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_b_count(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_found(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_A(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_B(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_steps_executed(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));

        // Upload initial frontiers and counters
        const int one = 1;
        const int zero = 0;
        pimpl->gpu.queue.enqueueWriteBuffer(buf_f_curr, CL_FALSE, 0, sizeof(int), &start_idx);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_b_curr, CL_FALSE, 0, sizeof(int), &end_idx);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_f_count, CL_FALSE, 0, sizeof(int), &one);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_b_count, CL_FALSE, 0, sizeof(int), &one);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_collision_found, CL_FALSE, 0, sizeof(int), &zero);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_steps_executed, CL_TRUE, 0, sizeof(int), &zero);

        int f_count = 1;
        int b_count = 1;
        total_frontier_expansions = 0;
        cells_visited = 2;
        int collision_found = 0;
        int cell_A = -1;
        int cell_B = -1;

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
            pimpl->gpu.queue.enqueueNDRangeKernel(
                pimpl->kernel_batched,
                cl::NullRange,
                cl::NDRange(LOCAL_WORKGROUP_SIZE),
                cl::NDRange(LOCAL_WORKGROUP_SIZE)
            );

            // Read collision status and counts once per batch
            pimpl->gpu.queue.enqueueReadBuffer(buf_collision_found, CL_FALSE, 0, sizeof(int), &collision_found);
            pimpl->gpu.queue.enqueueReadBuffer(buf_f_count, CL_FALSE, 0, sizeof(int), &f_count);
            pimpl->gpu.queue.enqueueReadBuffer(buf_b_count, CL_TRUE, 0, sizeof(int), &b_count);

            if (collision_found != 0) break;
            if (f_count == 0 || b_count == 0) break;
        }

        int steps_executed = 0;
        pimpl->gpu.queue.enqueueReadBuffer(buf_steps_executed, CL_TRUE, 0, sizeof(int), &steps_executed);
        total_frontier_expansions = steps_executed;

        if (collision_found == 0) {
            std::cerr << "[GpuBidirectionalBfsSolver] No path found between Start and End.\n";
            return false;
        }

        pimpl->gpu.queue.enqueueReadBuffer(buf_collision_cell_A, CL_FALSE, 0, sizeof(int), &cell_A);
        pimpl->gpu.queue.enqueueReadBuffer(buf_collision_cell_B, CL_FALSE, 0, sizeof(int), &cell_B);
        pimpl->gpu.queue.enqueueReadBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

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
