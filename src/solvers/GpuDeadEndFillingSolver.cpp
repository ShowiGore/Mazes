#include "GpuDeadEndFillingSolver.hpp"

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include "GpuUtils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <cstdlib>

static const char* EMBEDDED_DEAD_END_KERNELS = R"(
#define TILE_W 32
#define TILE_H 8
#define SH_W (TILE_W + 2)
#define SH_H (TILE_H + 2)

// =============================================================================
// KERNEL 1: 1-Step Coalesced Kernel with Shared Memory Halo
// =============================================================================
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
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    __local uchar s_tile[SH_H][SH_W];
    __local int l_changes;

    if (lx == 0 && ly == 0) {
        l_changes = 0;
    }

    s_tile[ly + 1][lx + 1] = (gy < height && gx < width) ? in_grid[gy * width + gx] : 1;

    if (ly == 0) {
        const int north_gy = gy - 1;
        s_tile[0][lx + 1] = (north_gy >= 0 && gx < width) ? in_grid[north_gy * width + gx] : 1;
    }
    if (ly == 7) {
        const int south_gy = gy + 1;
        s_tile[9][lx + 1] = (south_gy < height && gx < width) ? in_grid[south_gy * width + gx] : 1;
    }
    if (lx == 0) {
        const int west_gx = gx - 1;
        s_tile[ly + 1][0] = (gy < height && west_gx >= 0) ? in_grid[gy * width + west_gx] : 1;
    }
    if (lx == 31) {
        const int east_gx = gx + 1;
        s_tile[ly + 1][33] = (gy < height && east_gx < width) ? in_grid[gy * width + east_gx] : 1;
    }
    if (lx == 0 && ly == 0) {
        s_tile[0][0] = (gy > 0 && gx > 0) ? in_grid[(gy - 1) * width + (gx - 1)] : 1;
    }
    if (lx == 31 && ly == 0) {
        s_tile[0][33] = (gy > 0 && gx + 1 < width) ? in_grid[(gy - 1) * width + (gx + 1)] : 1;
    }
    if (lx == 0 && ly == 7) {
        s_tile[9][0] = (gy + 1 < height && gx > 0) ? in_grid[(gy + 1) * width + (gx - 1)] : 1;
    }
    if (lx == 31 && ly == 7) {
        s_tile[9][33] = (gy + 1 < height && gx + 1 < width) ? in_grid[(gy + 1) * width + (gx + 1)] : 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    bool pruned = false;

    if (gy < height && gx < width) {
        const int idx = gy * width + gx;
        const uchar cell = s_tile[ly + 1][lx + 1];

        if (cell != 0) {
            out_grid[idx] = 1;
        } else if ((gy == start_r && gx == start_c) || (gy == end_r && gx == end_c)) {
            out_grid[idx] = 0;
        } else {
            int open_count = 0;
            if (s_tile[ly][lx + 1] == 0) open_count++;
            if (s_tile[ly + 2][lx + 1] == 0) open_count++;
            if (s_tile[ly + 1][lx] == 0) open_count++;
            if (s_tile[ly + 1][lx + 2] == 0) open_count++;

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

    if (lx == 0 && ly == 0 && l_changes > 0) {
        atomic_add(change_count, l_changes);
    }
}

// =============================================================================
// KERNEL 2: 2-Step Sub-stepping Kernel (Halo R=2)
// =============================================================================
#define SUB_W (TILE_W + 4)
#define SUB_H (TILE_H + 4)

__kernel void dead_end_substep2(
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
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);
    const int tid = ly * TILE_W + lx;

    __local uchar s_grid0[SUB_H][SUB_W];
    __local uchar s_grid1[SUB_H][SUB_W];
    __local int l_changes;

    if (tid == 0) {
        l_changes = 0;
    }

    const int base_x = get_group_id(0) * TILE_W - 2;
    const int base_y = get_group_id(1) * TILE_H - 2;

    const int TOTAL_HALO = SUB_H * SUB_W;
    int idx0 = tid;
    if (idx0 < TOTAL_HALO) {
        int sy = idx0 / SUB_W;
        int sx = idx0 % SUB_W;
        int gy_in = base_y + sy;
        int gx_in = base_x + sx;
        s_grid0[sy][sx] = (gy_in >= 0 && gy_in < height && gx_in >= 0 && gx_in < width)
                          ? in_grid[gy_in * width + gx_in] : 1;
    }
    int idx1 = tid + 256;
    if (idx1 < TOTAL_HALO) {
        int sy = idx1 / SUB_W;
        int sx = idx1 % SUB_W;
        int gy_in = base_y + sy;
        int gx_in = base_x + sx;
        s_grid0[sy][sx] = (gy_in >= 0 && gy_in < height && gx_in >= 0 && gx_in < width)
                          ? in_grid[gy_in * width + gx_in] : 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int TOTAL_STEP1 = (SUB_H - 2) * (SUB_W - 2);
    bool pruned_any = false;

    if (tid < TOTAL_STEP1) {
        int sy = 1 + (tid / (SUB_W - 2));
        int sx = 1 + (tid % (SUB_W - 2));
        int cell_gy = base_y + sy;
        int cell_gx = base_x + sx;
        uchar cell = s_grid0[sy][sx];

        if (cell != 0) {
            s_grid1[sy][sx] = 1;
        } else if ((cell_gy == start_r && cell_gx == start_c) || (cell_gy == end_r && cell_gx == end_c)) {
            s_grid1[sy][sx] = 0;
        } else {
            int open_count = 0;
            if (s_grid0[sy - 1][sx] == 0) open_count++;
            if (s_grid0[sy + 1][sx] == 0) open_count++;
            if (s_grid0[sy][sx - 1] == 0) open_count++;
            if (s_grid0[sy][sx + 1] == 0) open_count++;

            if (open_count <= 1) {
                s_grid1[sy][sx] = 1;
                pruned_any = true;
            } else {
                s_grid1[sy][sx] = 0;
            }
        }
    }

    if (idx1 < TOTAL_STEP1) {
        int sy = 1 + (idx1 / (SUB_W - 2));
        int sx = 1 + (idx1 % (SUB_W - 2));
        int cell_gy = base_y + sy;
        int cell_gx = base_x + sx;
        uchar cell = s_grid0[sy][sx];

        if (cell != 0) {
            s_grid1[sy][sx] = 1;
        } else if ((cell_gy == start_r && cell_gx == start_c) || (cell_gy == end_r && cell_gx == end_c)) {
            s_grid1[sy][sx] = 0;
        } else {
            int open_count = 0;
            if (s_grid0[sy - 1][sx] == 0) open_count++;
            if (s_grid0[sy + 1][sx] == 0) open_count++;
            if (s_grid0[sy][sx - 1] == 0) open_count++;
            if (s_grid0[sy][sx + 1] == 0) open_count++;

            if (open_count <= 1) {
                s_grid1[sy][sx] = 1;
                pruned_any = true;
            } else {
                s_grid1[sy][sx] = 0;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gy < height && gx < width) {
        int sy = ly + 2;
        int sx = lx + 2;
        uchar cell = s_grid1[sy][sx];
        int g_idx = gy * width + gx;

        if (cell != 0) {
            out_grid[g_idx] = 1;
        } else if ((gy == start_r && gx == start_c) || (gy == end_r && gx == end_c)) {
            out_grid[g_idx] = 0;
        } else {
            int open_count = 0;
            if (s_grid1[sy - 1][sx] == 0) open_count++;
            if (s_grid1[sy + 1][sx] == 0) open_count++;
            if (s_grid1[sy][sx - 1] == 0) open_count++;
            if (s_grid1[sy][sx + 1] == 0) open_count++;

            if (open_count <= 1) {
                out_grid[g_idx] = 1;
                pruned_any = true;
            } else {
                out_grid[g_idx] = 0;
            }
        }
    }

    if (pruned_any) {
        atomic_inc(&l_changes);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0 && l_changes > 0) {
        atomic_add(change_count, l_changes);
    }
}

// =============================================================================
// KERNEL 3: Bit-Packed 32-Cell SIMD Kernel
// =============================================================================
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

    if (r >= height || col_word >= words_per_row) {
        return;
    }

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

    // 4-bitplane parallel full adder
    const uint sum1 = w0 ^ e0;
    const uint c1   = w0 & e0;

    const uint sum2 = sum1 ^ n0;
    const uint c2   = (sum1 & n0) | c1;

    const uint carry = (sum2 & s0) | c2; // bit is 1 iff open_neighbors >= 2

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
)";

struct GpuDeadEndFillingSolver::Impl {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel_coalesced_A;
    cl::Kernel kernel_coalesced_B;
    cl::Kernel kernel_substep_A;
    cl::Kernel kernel_substep_B;
    cl::Kernel kernel_bitpacked_A;
    cl::Kernel kernel_bitpacked_B;
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
            std::cerr << "[GpuDeadEndFillingSolver] Error: No suitable OpenCL device found.\n";
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
            kernel_source = EMBEDDED_DEAD_END_KERNELS;
        }

        cl::Program::Sources sources = {{kernel_source.c_str(), kernel_source.length()}};
        pimpl->program = cl::Program(pimpl->context, sources);

        if (pimpl->program.build({pimpl->device}, "-cl-std=CL2.0 -cl-fast-relaxed-math") != CL_SUCCESS) {
            std::string log = pimpl->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(pimpl->device);
            std::cerr << "[GpuDeadEndFillingSolver] Kernel compilation failed:\n" << log << std::endl;
            return false;
        }

        pimpl->kernel_coalesced_A = cl::Kernel(pimpl->program, "dead_end_filling_step");
        pimpl->kernel_coalesced_B = cl::Kernel(pimpl->program, "dead_end_filling_step");

        pimpl->kernel_substep_A = cl::Kernel(pimpl->program, "dead_end_substep2");
        pimpl->kernel_substep_B = cl::Kernel(pimpl->program, "dead_end_substep2");

        pimpl->kernel_bitpacked_A = cl::Kernel(pimpl->program, "dead_end_bitpacked_step");
        pimpl->kernel_bitpacked_B = cl::Kernel(pimpl->program, "dead_end_bitpacked_step");

        pimpl->ready = true;
        is_initialized = true;

        std::cout << "[GpuDeadEndFillingSolver] OpenCL Initialized on "
                  << selected_device_name << " (" << selected_platform_name << ")\n";
        return true;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuDeadEndFillingSolver] OpenCL Exception: " << err.what()
                  << " (" << err.err() << ")\n";
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

    // =========================================================================
    // MODE 1: BITPACKED (32 cells per uint word - 8x less VRAM, fastest)
    // =========================================================================
    if (this->mode == GpuDeadEndMode::BITPACKED) {
        const int words_per_row = (this->width + 31) / 32;
        const size_t total_words = static_cast<size_t>(this->height) * words_per_row;

        std::vector<uint32_t> host_words(total_words, 0xFFFFFFFFU);
        for (int r = 0; r < this->height; ++r) {
            const size_t row_offset = static_cast<size_t>(r) * words_per_row;
            for (int c = 0; c < this->width; ++c) {
                const int word_idx = c / 32;
                const int bit_idx = c % 32;
                if (!grid[r][c]) { // PATH = 0
                    host_words[row_offset + word_idx] &= ~(1U << bit_idx);
                }
            }
        }

        try {
            cl::Buffer buf_in(pimpl->context, CL_MEM_READ_WRITE, total_words * sizeof(uint32_t));
            cl::Buffer buf_out(pimpl->context, CL_MEM_READ_WRITE, total_words * sizeof(uint32_t));
            cl::Buffer buf_changes(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

            pimpl->queue.enqueueWriteBuffer(buf_in, CL_TRUE, 0, total_words * sizeof(uint32_t), host_words.data());

            constexpr size_t TILE_W = 32;
            constexpr size_t TILE_H = 8;
            const cl::NDRange local_range(TILE_W, TILE_H);
            const cl::NDRange global_range(
                ((words_per_row + TILE_W - 1) / TILE_W) * TILE_W,
                ((this->height + TILE_H - 1) / TILE_H) * TILE_H
            );

            pimpl->kernel_bitpacked_A.setArg(0, buf_in);
            pimpl->kernel_bitpacked_A.setArg(1, buf_out);
            pimpl->kernel_bitpacked_A.setArg(2, buf_changes);
            pimpl->kernel_bitpacked_A.setArg(3, this->height);
            pimpl->kernel_bitpacked_A.setArg(4, words_per_row);
            pimpl->kernel_bitpacked_A.setArg(5, this->width);
            pimpl->kernel_bitpacked_A.setArg(6, this->start.first);
            pimpl->kernel_bitpacked_A.setArg(7, this->start.second);
            pimpl->kernel_bitpacked_A.setArg(8, this->end.first);
            pimpl->kernel_bitpacked_A.setArg(9, this->end.second);

            pimpl->kernel_bitpacked_B.setArg(0, buf_out);
            pimpl->kernel_bitpacked_B.setArg(1, buf_in);
            pimpl->kernel_bitpacked_B.setArg(2, buf_changes);
            pimpl->kernel_bitpacked_B.setArg(3, this->height);
            pimpl->kernel_bitpacked_B.setArg(4, words_per_row);
            pimpl->kernel_bitpacked_B.setArg(5, this->width);
            pimpl->kernel_bitpacked_B.setArg(6, this->start.first);
            pimpl->kernel_bitpacked_B.setArg(7, this->start.second);
            pimpl->kernel_bitpacked_B.setArg(8, this->end.first);
            pimpl->kernel_bitpacked_B.setArg(9, this->end.second);

            // Compute adaptive GPU execution parameters
            const auto adaptive_cfg = computeGpuAdaptiveConfig(
                pimpl->device,
                this->height,
                this->width,
                1, // bitpacked is ~0.125 bytes per cell
                this->user_batch_size
            );
            this->config_rationale = adaptive_cfg.rationale;
            const int BATCH_SIZE = (adaptive_cfg.batch_size / 2) * 2; // ensure even for ping-pong

            total_iterations = 0;
            int changes = 0;
            const int zero = 0;

            while (true) {
                pimpl->queue.enqueueWriteBuffer(buf_changes, CL_FALSE, 0, sizeof(int), &zero);

                for (int b = 0; b < BATCH_SIZE; b += 2) {
                    pimpl->queue.enqueueNDRangeKernel(pimpl->kernel_bitpacked_A, cl::NullRange, global_range, local_range);
                    pimpl->queue.enqueueNDRangeKernel(pimpl->kernel_bitpacked_B, cl::NullRange, global_range, local_range);
                    total_iterations += 2;
                }

                pimpl->queue.enqueueReadBuffer(buf_changes, CL_TRUE, 0, sizeof(int), &changes);

                if (changes == 0) {
                    break;
                }
            }

            pimpl->queue.enqueueReadBuffer(buf_in, CL_TRUE, 0, total_words * sizeof(uint32_t), host_words.data());

            this->solution.assign(this->height, std::vector<bool>(this->width, false));
            this->visited.assign(this->height, std::vector<bool>(this->width, false));

            size_t solution_cells = 0;
            for (int r = 0; r < this->height; ++r) {
                const size_t row_offset = static_cast<size_t>(r) * words_per_row;
                for (int c = 0; c < this->width; ++c) {
                    const int word_idx = c / 32;
                    const int bit_idx = c % 32;
                    const bool original_is_path = !grid[r][c];
                    const bool surviving_is_path = ((host_words[row_offset + word_idx] & (1U << bit_idx)) == 0);

                    if (surviving_is_path) {
                        this->solution[r][c] = true;
                        this->visited[r][c] = true;
                        solution_cells++;
                    } else if (original_is_path) {
                        this->visited[r][c] = true;
                    }
                }
            }

            std::cout << "[GpuDeadEndFillingSolver] (BITPACKED) Converged in " << total_iterations
                      << " parallel GPU steps (" << solution_cells << " solution cells)\n";

            return (solution_cells > 0 &&
                    this->solution[this->start.first][this->start.second] &&
                    this->solution[this->end.first][this->end.second]);

        } catch (const cl::Error &err) {
            std::cerr << "[GpuDeadEndFillingSolver] OpenCL error: " << err.what() << " (" << err.err() << ")\n";
            return false;
        }
    }

    // =========================================================================
    // MODE 2: SUBSTEPPING / COALESCED (1 byte per cell)
    // =========================================================================
    const size_t total_cells = static_cast<size_t>(this->height) * this->width;
    std::vector<uint8_t> host_grid(total_cells);
    for (int r = 0; r < this->height; ++r) {
        const size_t row_offset = static_cast<size_t>(r) * this->width;
        for (int c = 0; c < this->width; ++c) {
            host_grid[row_offset + c] = grid[r][c] ? 1 : 0;
        }
    }

    try {
        cl::Buffer buf_in(pimpl->context, CL_MEM_READ_WRITE, total_cells * sizeof(uint8_t));
        cl::Buffer buf_out(pimpl->context, CL_MEM_READ_WRITE, total_cells * sizeof(uint8_t));
        cl::Buffer buf_changes(pimpl->context, CL_MEM_READ_WRITE, sizeof(int));

        pimpl->queue.enqueueWriteBuffer(buf_in, CL_TRUE, 0, total_cells * sizeof(uint8_t), host_grid.data());

        constexpr size_t TILE_W = 32;
        constexpr size_t TILE_H = 8;
        const cl::NDRange local_range(TILE_W, TILE_H);
        const cl::NDRange global_range(
            ((this->width + TILE_W - 1) / TILE_W) * TILE_W,
            ((this->height + TILE_H - 1) / TILE_H) * TILE_H
        );

        cl::Kernel &k_A = (this->mode == GpuDeadEndMode::SUBSTEPPING) ? pimpl->kernel_substep_A : pimpl->kernel_coalesced_A;
        cl::Kernel &k_B = (this->mode == GpuDeadEndMode::SUBSTEPPING) ? pimpl->kernel_substep_B : pimpl->kernel_coalesced_B;
        const int step_multiplier = (this->mode == GpuDeadEndMode::SUBSTEPPING) ? 2 : 1;

        k_A.setArg(0, buf_in);
        k_A.setArg(1, buf_out);
        k_A.setArg(2, buf_changes);
        k_A.setArg(3, this->height);
        k_A.setArg(4, this->width);
        k_A.setArg(5, this->start.first);
        k_A.setArg(6, this->start.second);
        k_A.setArg(7, this->end.first);
        k_A.setArg(8, this->end.second);

        k_B.setArg(0, buf_out);
        k_B.setArg(1, buf_in);
        k_B.setArg(2, buf_changes);
        k_B.setArg(3, this->height);
        k_B.setArg(4, this->width);
        k_B.setArg(5, this->start.first);
        k_B.setArg(6, this->start.second);
        k_B.setArg(7, this->end.first);
        k_B.setArg(8, this->end.second);

        total_iterations = 0;
        int changes = 0;
        const int zero = 0;
        constexpr int BATCH_SIZE = 128;

        while (true) {
            pimpl->queue.enqueueWriteBuffer(buf_changes, CL_FALSE, 0, sizeof(int), &zero);

            for (int b = 0; b < BATCH_SIZE; b += 2) {
                pimpl->queue.enqueueNDRangeKernel(k_A, cl::NullRange, global_range, local_range);
                pimpl->queue.enqueueNDRangeKernel(k_B, cl::NullRange, global_range, local_range);
                total_iterations += 2 * step_multiplier;
            }

            pimpl->queue.enqueueReadBuffer(buf_changes, CL_TRUE, 0, sizeof(int), &changes);

            if (changes == 0) {
                break;
            }
        }

        pimpl->queue.enqueueReadBuffer(buf_in, CL_TRUE, 0, total_cells * sizeof(uint8_t), host_grid.data());

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
                    this->visited[r][c] = true;
                }
            }
        }

        std::cout << "[GpuDeadEndFillingSolver] ("
                  << (this->mode == GpuDeadEndMode::SUBSTEPPING ? "SUBSTEPPING" : "COALESCED")
                  << ") Converged in " << total_iterations
                  << " parallel GPU steps (" << solution_cells << " solution cells)\n";

        return (solution_cells > 0 &&
                this->solution[this->start.first][this->start.second] &&
                this->solution[this->end.first][this->end.second]);

    } catch (const cl::Error &err) {
        std::cerr << "[GpuDeadEndFillingSolver] OpenCL error: " << err.what() << " (" << err.err() << ")\n";
        return false;
    }
}
