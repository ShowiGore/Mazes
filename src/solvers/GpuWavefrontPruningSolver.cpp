#include "GpuWavefrontPruningSolver.hpp"

#include "utilities/GpuUtils.hpp"

#include <iostream>
#include <vector>
#include <algorithm>

static const char* EMBEDDED_WAVEFRONT_KERNELS = R"(
__kernel void dead_end_bitpacked_step(
    __global const uint* in_words,
    __global uint* out_words,
    __global int* change_count,
    const int height,
    const int words_per_row,
    const int width,
    const int start_r,
    const int start_c,
    const int end_r,
    const int end_c
) {
    const int col_word = get_global_id(0);
    const int r = get_global_id(1);

    if (r >= height || col_word >= words_per_row) return;

    const int idx = r * words_per_row + col_word;
    const uint curr = in_words[idx];

    if (curr == 0xFFFFFFFFU) {
        out_words[idx] = 0xFFFFFFFFU;
        return;
    }

    const uint north = (r > 0) ? in_words[(r - 1) * words_per_row + col_word] : 0xFFFFFFFFU;
    const uint south = (r + 1 < height) ? in_words[(r + 1) * words_per_row + col_word] : 0xFFFFFFFFU;

    const uint west_word = (col_word > 0) ? in_words[r * words_per_row + (col_word - 1)] : 0xFFFFFFFFU;
    const uint east_word = (col_word + 1 < words_per_row) ? in_words[r * words_per_row + (col_word + 1)] : 0xFFFFFFFFU;

    const uint west = (curr << 1) | (west_word >> 31);
    const uint east = (curr >> 1) | (east_word << 31);

    const uint w0 = ~west;
    const uint e0 = ~east;
    const uint n0 = ~north;
    const uint s0 = ~south;

    const uint sum1 = w0 ^ e0;
    const uint c1   = w0 & e0;
    const uint sum2 = sum1 ^ n0;
    const uint c2   = (sum1 & n0) | c1;
    const uint carry = (sum2 & s0) | c2;

    uint prune_mask = (~curr) & (~carry);

    if (r == start_r && (start_c / 32) == col_word) {
        prune_mask &= ~(1U << (start_c % 32));
    }
    if (r == end_r && (end_c / 32) == col_word) {
        prune_mask &= ~(1U << (end_c % 32));
    }

    if (col_word == words_per_row - 1 && (width % 32) != 0) {
        const uint valid_bits = (1U << (width % 32)) - 1U;
        prune_mask &= valid_bits;
    }

    if (prune_mask != 0) {
        atomic_add(change_count, popcount(prune_mask));
    }

    out_words[idx] = curr | prune_mask;
}

__kernel void wavefront_expand_step(
    __global const uint* in_grid,
    __global uint* visited_self,
    __global const uint* visited_other,
    __global uint* parent_dir,
    __global const int2* curr_frontier,
    const int curr_count,
    __global int2* next_frontier,
    __global int* next_count,
    const int max_frontier_size,
    __global int* collision_flag,
    __global int2* collision_cell_self,
    __global int2* collision_cell_other,
    const int height,
    const int width,
    const int words_per_row,
    const int dir_words_per_row
) {
    const int tid = get_global_id(0);
    if (tid >= curr_count) return;

    const int2 curr = curr_frontier[tid];
    const int r = curr.x;
    const int c = curr.y;

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    const uint opp_dir[4] = {1U, 0U, 3U, 2U};

    for (int i = 0; i < 4; ++i) {
        const int nr = r + dr[i];
        const int nc = c + dc[i];

        if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;

        const int g_idx = nr * words_per_row + (nc / 32);
        const uint bit = 1U << (nc % 32);

        if ((in_grid[g_idx] & bit) != 0) continue;

        const uint prev_visited = atomic_or(&visited_self[g_idx], bit);
        if ((prev_visited & bit) != 0) continue;

        const int d_idx = nr * dir_words_per_row + (nc / 16);
        const uint d_shift = (nc % 16) * 2;
        atomic_or(&parent_dir[d_idx], (opp_dir[i] & 3U) << d_shift);

        if ((visited_other[g_idx] & bit) != 0) {
            atomic_xchg(collision_flag, 1);
            *collision_cell_self = (int2)(nr, nc);
            *collision_cell_other = (int2)(nr, nc);
        }

        const int pos = atomic_inc(next_count);
        if (pos < max_frontier_size) {
            next_frontier[pos] = (int2)(nr, nc);
        }
    }
}

__kernel void trace_path(
    __global const uint* parent_dir,
    const int2 target_cell,
    const int2 meet_cell,
    __global int2* out_path,
    __global int* out_length,
    const int max_path_cells,
    const int dir_words_per_row
) {
    if (get_global_id(0) != 0) return;

    int r = meet_cell.x;
    int c = meet_cell.y;
    int len = 0;

    if (len < max_path_cells) {
        out_path[len++] = (int2)(r, c);
    }

    while (r != target_cell.x || c != target_cell.y) {
        const int d_idx = r * dir_words_per_row + (c / 16);
        const uint d_shift = (c % 16) * 2;
        const uint dir = (parent_dir[d_idx] >> d_shift) & 3U;

        if (dir == 0)      r -= 1;
        else if (dir == 1) r += 1;
        else if (dir == 2) c -= 1;
        else if (dir == 3) c += 1;

        if (len < max_path_cells) {
            out_path[len++] = (int2)(r, c);
        } else {
            break;
        }
    }

    *out_length = len;
}
)";

struct GpuWavefrontPruningSolver::Impl {
    GpuContext gpu;
    cl::Program program;
    cl::Kernel kernel_prune;
    cl::Kernel kernel_wavefront;
    cl::Kernel kernel_trace;
};

GpuWavefrontPruningSolver::GpuWavefrontPruningSolver(std::string device_vendor)
    : GpuSolver(std::move(device_vendor)),
      pimpl(std::make_unique<Impl>()) {
    this->solver_name = "gpu-wavefront";
}

GpuWavefrontPruningSolver::~GpuWavefrontPruningSolver() = default;

GpuWavefrontPruningSolver::GpuWavefrontPruningSolver(GpuWavefrontPruningSolver&&) noexcept = default;
GpuWavefrontPruningSolver& GpuWavefrontPruningSolver::operator=(GpuWavefrontPruningSolver&&) noexcept = default;

bool GpuWavefrontPruningSolver::ensureOpenCLInitialized() {
    if (pimpl->gpu.ready) return true;

    if (!pimpl->gpu.init(this->preferred_device_vendor, "[GpuWavefrontPruningSolver]")) {
        return false;
    }
    this->selected_platform_name = pimpl->gpu.platform_name;
    this->selected_device_name = pimpl->gpu.device_name;

    const std::string kernel_source = GpuContext::loadKernelSource("wavefront_kernels.cl", EMBEDDED_WAVEFRONT_KERNELS);
    pimpl->program = pimpl->gpu.buildProgram(kernel_source, "-cl-std=CL2.0 -cl-fast-relaxed-math", "[GpuWavefrontPruningSolver]");
    if (!pimpl->gpu.ready) return false;

    pimpl->kernel_prune = cl::Kernel(pimpl->program, "dead_end_bitpacked_step");
    pimpl->kernel_wavefront = cl::Kernel(pimpl->program, "wavefront_expand_step");
    pimpl->kernel_trace = cl::Kernel(pimpl->program, "trace_path");
    return true;
}

bool GpuWavefrontPruningSolver::solve(const Maze &maze) {
    if (!ensureOpenCLInitialized()) {
        std::cerr << "[GpuWavefrontPruningSolver] Failed to initialize OpenCL runtime.\n";
        return false;
    }

    this->height = maze.getHeight();
    this->width = maze.getWidth();
    this->start = maze.getStart();
    this->end = maze.getEnd();

    const int H = this->height;
    const int W = this->width;
    const int words_per_row = (W + 31) / 32;
    const size_t total_words = static_cast<size_t>(H) * words_per_row;
    const size_t total_grid_bytes = total_words * sizeof(uint32_t);

    const int dir_words_per_row = (W + 15) / 16;
    const size_t total_dir_words = static_cast<size_t>(H) * dir_words_per_row;
    const size_t total_dir_bytes = total_dir_words * sizeof(uint32_t);

    try {
        // =====================================================================
        // STEP 1: Bitpack Maze Grid into 32-bit Words (1 = WALL, 0 = PATH)
        // =====================================================================
        const auto &maze_grid = maze.getMaze();
        std::vector<uint32_t> host_grid(total_words, 0xFFFFFFFFU);

        for (int r = 0; r < H; ++r) {
            const size_t row_offset = static_cast<size_t>(r) * words_per_row;
            for (int c = 0; c < W; ++c) {
                if (!maze_grid[r][c]) {
                    host_grid[row_offset + (c / 32)] &= ~(1U << (c % 32));
                }
            }
        }

        cl::Buffer buf_grid_A(pimpl->gpu.context, CL_MEM_READ_WRITE, total_grid_bytes);
        cl::Buffer buf_grid_B(pimpl->gpu.context, CL_MEM_READ_WRITE, total_grid_bytes);
        cl::Buffer buf_changes(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));

        pimpl->gpu.queue.enqueueWriteBuffer(buf_grid_A, CL_TRUE, 0, total_grid_bytes, host_grid.data());

        // =====================================================================
        // STEP 2: Phase 1 - Asynchronous Bitpacked Dead-End Pruning in VRAM
        // =====================================================================
        constexpr size_t TILE_X = 16;
        constexpr size_t TILE_Y = 16;
        const cl::NDRange prune_local(TILE_X, TILE_Y);
        const cl::NDRange prune_global(
            ((words_per_row + TILE_X - 1) / TILE_X) * TILE_X,
            ((H + TILE_Y - 1) / TILE_Y) * TILE_Y
        );

        pimpl->kernel_prune.setArg(3, H);
        pimpl->kernel_prune.setArg(4, words_per_row);
        pimpl->kernel_prune.setArg(5, W);
        pimpl->kernel_prune.setArg(6, this->start.first);
        pimpl->kernel_prune.setArg(7, this->start.second);
        pimpl->kernel_prune.setArg(8, this->end.first);
        pimpl->kernel_prune.setArg(9, this->end.second);

        this->prune_passes = 0;
        this->total_pruned = 0;
        cl::Buffer *cur_in = &buf_grid_A;
        cl::Buffer *cur_out = &buf_grid_B;

        const int batch_passes = (this->user_batch_size > 0) ? this->user_batch_size : 32;

        for (int pass = 0; pass < max_prune_iterations; pass += batch_passes) {
            const int zero = 0;
            pimpl->gpu.queue.enqueueWriteBuffer(buf_changes, CL_FALSE, 0, sizeof(int), &zero);

            const int passes_to_run = std::min(batch_passes, max_prune_iterations - pass);
            for (int b = 0; b < passes_to_run; ++b) {
                pimpl->kernel_prune.setArg(0, *cur_in);
                pimpl->kernel_prune.setArg(1, *cur_out);
                pimpl->kernel_prune.setArg(2, buf_changes);

                pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_prune, cl::NullRange, prune_global, prune_local);
                std::swap(cur_in, cur_out);
                this->prune_passes++;
            }

            int changes = 0;
            pimpl->gpu.queue.enqueueReadBuffer(buf_changes, CL_TRUE, 0, sizeof(int), &changes);
            this->total_pruned += changes;

            if (changes == 0) break;
        }

        // cur_in holds the final pruned maze buffer; release the other buffer to free VRAM
        cl::Buffer buf_pruned = *cur_in;
        *cur_out = cl::Buffer();
        if (cur_in == &buf_grid_A) {
            buf_grid_B = cl::Buffer();
        } else {
            buf_grid_A = cl::Buffer();
        }

        // =====================================================================
        // STEP 3: Phase 2 - Dual-Frontier Bitmapped Wavefront Expansion
        // =====================================================================
        cl::Buffer buf_visited_fwd(pimpl->gpu.context, CL_MEM_READ_WRITE, total_grid_bytes);
        cl::Buffer buf_visited_bwd(pimpl->gpu.context, CL_MEM_READ_WRITE, total_grid_bytes);
        cl::Buffer buf_parent_fwd(pimpl->gpu.context, CL_MEM_READ_WRITE, total_dir_bytes);
        cl::Buffer buf_parent_bwd(pimpl->gpu.context, CL_MEM_READ_WRITE, total_dir_bytes);

        const uint32_t zero_u32 = 0;
        pimpl->gpu.queue.enqueueFillBuffer(buf_visited_fwd, zero_u32, 0, total_grid_bytes);
        pimpl->gpu.queue.enqueueFillBuffer(buf_visited_bwd, zero_u32, 0, total_grid_bytes);
        pimpl->gpu.queue.enqueueFillBuffer(buf_parent_fwd, zero_u32, 0, total_dir_bytes);
        pimpl->gpu.queue.enqueueFillBuffer(buf_parent_bwd, zero_u32, 0, total_dir_bytes);

        // Frontier queues in VRAM
        constexpr int MAX_FRONTIER = 524288;
        cl::Buffer buf_fwd_curr(pimpl->gpu.context, CL_MEM_READ_WRITE, MAX_FRONTIER * sizeof(cl_int2));
        cl::Buffer buf_fwd_next(pimpl->gpu.context, CL_MEM_READ_WRITE, MAX_FRONTIER * sizeof(cl_int2));
        cl::Buffer buf_bwd_curr(pimpl->gpu.context, CL_MEM_READ_WRITE, MAX_FRONTIER * sizeof(cl_int2));
        cl::Buffer buf_bwd_next(pimpl->gpu.context, CL_MEM_READ_WRITE, MAX_FRONTIER * sizeof(cl_int2));

        cl::Buffer buf_fwd_count(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_bwd_count(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_flag(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_collision_cell_self(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(cl_int2));
        cl::Buffer buf_collision_cell_other(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(cl_int2));

        // Seed initial frontiers
        cl_int2 start_coord = {this->start.first, this->start.second};
        cl_int2 end_coord = {this->end.first, this->end.second};
        pimpl->gpu.queue.enqueueWriteBuffer(buf_fwd_curr, CL_FALSE, 0, sizeof(cl_int2), &start_coord);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_bwd_curr, CL_FALSE, 0, sizeof(cl_int2), &end_coord);

        // Mark start & end in visited
        const size_t s_idx = static_cast<size_t>(this->start.first) * words_per_row + (this->start.second / 32);
        const uint32_t s_bit = 1U << (this->start.second % 32);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_visited_fwd, CL_FALSE, s_idx * sizeof(uint32_t), sizeof(uint32_t), &s_bit);

        const size_t e_idx = static_cast<size_t>(this->end.first) * words_per_row + (this->end.second / 32);
        const uint32_t e_bit = 1U << (this->end.second % 32);
        pimpl->gpu.queue.enqueueWriteBuffer(buf_visited_bwd, CL_FALSE, e_idx * sizeof(uint32_t), sizeof(uint32_t), &e_bit);

        const int zero_int = 0;
        pimpl->gpu.queue.enqueueWriteBuffer(buf_collision_flag, CL_TRUE, 0, sizeof(int), &zero_int);

        int fwd_count = 1;
        int bwd_count = 1;
        this->expansions_count = 0;
        this->visited_count = 2;

        cl::Buffer *cur_fwd_in = &buf_fwd_curr;
        cl::Buffer *cur_fwd_out = &buf_fwd_next;
        cl::Buffer *cur_bwd_in = &buf_bwd_curr;
        cl::Buffer *cur_bwd_out = &buf_bwd_next;

        int collision_detected = 0;
        cl_int2 meet_cell = {0, 0};

        while (fwd_count > 0 && bwd_count > 0 && !collision_detected) {
            this->expansions_count++;

            // Reset next counts
            pimpl->gpu.queue.enqueueWriteBuffer(buf_fwd_count, CL_FALSE, 0, sizeof(int), &zero_int);
            pimpl->gpu.queue.enqueueWriteBuffer(buf_bwd_count, CL_FALSE, 0, sizeof(int), &zero_int);

            // --- Expand Forward Frontier ---
            {
                const size_t local_sz = 64;
                const size_t global_sz = ((fwd_count + local_sz - 1) / local_sz) * local_sz;

                pimpl->kernel_wavefront.setArg(0, buf_pruned);
                pimpl->kernel_wavefront.setArg(1, buf_visited_fwd);
                pimpl->kernel_wavefront.setArg(2, buf_visited_bwd);
                pimpl->kernel_wavefront.setArg(3, buf_parent_fwd);
                pimpl->kernel_wavefront.setArg(4, *cur_fwd_in);
                pimpl->kernel_wavefront.setArg(5, fwd_count);
                pimpl->kernel_wavefront.setArg(6, *cur_fwd_out);
                pimpl->kernel_wavefront.setArg(7, buf_fwd_count);
                pimpl->kernel_wavefront.setArg(8, MAX_FRONTIER);
                pimpl->kernel_wavefront.setArg(9, buf_collision_flag);
                pimpl->kernel_wavefront.setArg(10, buf_collision_cell_self);
                pimpl->kernel_wavefront.setArg(11, buf_collision_cell_other);
                pimpl->kernel_wavefront.setArg(12, H);
                pimpl->kernel_wavefront.setArg(13, W);
                pimpl->kernel_wavefront.setArg(14, words_per_row);
                pimpl->kernel_wavefront.setArg(15, dir_words_per_row);

                pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_wavefront, cl::NullRange,
                                                      cl::NDRange(global_sz), cl::NDRange(local_sz));
            }

            // --- Expand Backward Frontier ---
            {
                const size_t local_sz = 64;
                const size_t global_sz = ((bwd_count + local_sz - 1) / local_sz) * local_sz;

                pimpl->kernel_wavefront.setArg(0, buf_pruned);
                pimpl->kernel_wavefront.setArg(1, buf_visited_bwd);
                pimpl->kernel_wavefront.setArg(2, buf_visited_fwd);
                pimpl->kernel_wavefront.setArg(3, buf_parent_bwd);
                pimpl->kernel_wavefront.setArg(4, *cur_bwd_in);
                pimpl->kernel_wavefront.setArg(5, bwd_count);
                pimpl->kernel_wavefront.setArg(6, *cur_bwd_out);
                pimpl->kernel_wavefront.setArg(7, buf_bwd_count);
                pimpl->kernel_wavefront.setArg(8, MAX_FRONTIER);
                pimpl->kernel_wavefront.setArg(9, buf_collision_flag);
                pimpl->kernel_wavefront.setArg(10, buf_collision_cell_self);
                pimpl->kernel_wavefront.setArg(11, buf_collision_cell_other);
                pimpl->kernel_wavefront.setArg(12, H);
                pimpl->kernel_wavefront.setArg(13, W);
                pimpl->kernel_wavefront.setArg(14, words_per_row);
                pimpl->kernel_wavefront.setArg(15, dir_words_per_row);

                pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_wavefront, cl::NullRange,
                                                      cl::NDRange(global_sz), cl::NDRange(local_sz));
            }

            // Read counts and collision flag
            pimpl->gpu.queue.enqueueReadBuffer(buf_collision_flag, CL_FALSE, 0, sizeof(int), &collision_detected);
            pimpl->gpu.queue.enqueueReadBuffer(buf_fwd_count, CL_FALSE, 0, sizeof(int), &fwd_count);
            pimpl->gpu.queue.enqueueReadBuffer(buf_bwd_count, CL_TRUE, 0, sizeof(int), &bwd_count);

            this->visited_count += fwd_count + bwd_count;

            if (collision_detected) {
                pimpl->gpu.queue.enqueueReadBuffer(buf_collision_cell_self, CL_TRUE, 0, sizeof(cl_int2), &meet_cell);
                break;
            }

            std::swap(cur_fwd_in, cur_fwd_out);
            std::swap(cur_bwd_in, cur_bwd_out);
        }

        if (!collision_detected) {
            std::cerr << "[GpuWavefrontPruningSolver] Wavefront exhausted without collision!\n";
            return false;
        }

        // =====================================================================
        // STEP 4: Phase 3 - In-VRAM Path Reconstruction
        // =====================================================================
        constexpr int MAX_PATH_CELLS = 4194304;
        cl::Buffer buf_path_fwd(pimpl->gpu.context, CL_MEM_READ_WRITE, MAX_PATH_CELLS * sizeof(cl_int2));
        cl::Buffer buf_path_bwd(pimpl->gpu.context, CL_MEM_READ_WRITE, MAX_PATH_CELLS * sizeof(cl_int2));
        cl::Buffer buf_len_fwd(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));
        cl::Buffer buf_len_bwd(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));

        // Trace forward from meet_cell back to start
        pimpl->kernel_trace.setArg(0, buf_parent_fwd);
        pimpl->kernel_trace.setArg(1, start_coord);
        pimpl->kernel_trace.setArg(2, meet_cell);
        pimpl->kernel_trace.setArg(3, buf_path_fwd);
        pimpl->kernel_trace.setArg(4, buf_len_fwd);
        pimpl->kernel_trace.setArg(5, MAX_PATH_CELLS);
        pimpl->kernel_trace.setArg(6, dir_words_per_row);
        pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_trace, cl::NullRange, cl::NDRange(1), cl::NDRange(1));

        // Trace backward from meet_cell back to end
        pimpl->kernel_trace.setArg(0, buf_parent_bwd);
        pimpl->kernel_trace.setArg(1, end_coord);
        pimpl->kernel_trace.setArg(2, meet_cell);
        pimpl->kernel_trace.setArg(3, buf_path_bwd);
        pimpl->kernel_trace.setArg(4, buf_len_bwd);
        pimpl->kernel_trace.setArg(5, MAX_PATH_CELLS);
        pimpl->kernel_trace.setArg(6, dir_words_per_row);
        pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_trace, cl::NullRange, cl::NDRange(1), cl::NDRange(1));

        int len_fwd = 0;
        int len_bwd = 0;
        pimpl->gpu.queue.enqueueReadBuffer(buf_len_fwd, CL_FALSE, 0, sizeof(int), &len_fwd);
        pimpl->gpu.queue.enqueueReadBuffer(buf_len_bwd, CL_TRUE, 0, sizeof(int), &len_bwd);

        std::vector<cl_int2> host_path_fwd(len_fwd);
        std::vector<cl_int2> host_path_bwd(len_bwd);
        pimpl->gpu.queue.enqueueReadBuffer(buf_path_fwd, CL_FALSE, 0, len_fwd * sizeof(cl_int2), host_path_fwd.data());
        pimpl->gpu.queue.enqueueReadBuffer(buf_path_bwd, CL_TRUE, 0, len_bwd * sizeof(cl_int2), host_path_bwd.data());

        this->solution.assign(H, std::vector<bool>(W, false));
        this->visited.assign(H, std::vector<bool>(W, false));

        size_t solution_cells = 0;
        for (const auto &p : host_path_fwd) {
            if (!this->solution[p.x][p.y]) {
                this->solution[p.x][p.y] = true;
                this->visited[p.x][p.y] = true;
                solution_cells++;
            }
        }
        for (const auto &p : host_path_bwd) {
            if (!this->solution[p.x][p.y]) {
                this->solution[p.x][p.y] = true;
                this->visited[p.x][p.y] = true;
                solution_cells++;
            }
        }

        std::cout << "[GpuWavefrontPruningSolver] Pruned " << this->total_pruned
                  << " dead ends in " << this->prune_passes << " passes. "
                  << "Wavefront reached collision at (" << meet_cell.x << ", " << meet_cell.y
                  << ") in " << this->expansions_count << " steps (" << solution_cells << " solution cells).\n";

        return (solution_cells > 0 &&
                this->solution[this->start.first][this->start.second] &&
                this->solution[this->end.first][this->end.second]);

    } catch (const cl::Error &err) {
        std::cerr << "[GpuWavefrontPruningSolver] OpenCL error: " << err.what() << " (" << err.err() << ")\n";
        return false;
    }
}
