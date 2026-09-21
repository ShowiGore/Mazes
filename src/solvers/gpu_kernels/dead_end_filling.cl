/**
 * @file dead_end_filling.cl
 * @brief Ultra-Optimized OpenCL Kernels for Parallel Dead-End Filling
 *
 * Contains three specialized kernels:
 * 1. dead_end_filling_step:
 *    Natural coalesced 32x8 tile with 1-cell shared memory halo staging.
 *
 * 2. dead_end_substep2:
 *    Shared memory sub-stepping (halo R=2). Computes 2 cellular automaton
 *    steps entirely in on-chip SRAM per global VRAM load/store, cutting
 *    global memory traffic in half.
 *
 * 3. dead_end_bitpacked_step:
 *    Bit-packed 32-cell SIMD kernel. Each 32-bit word represents 32 horizontal
 *    cells. Evaluates all 32 cells simultaneously using parallel bitplane full-adders
 *    in 15 bitwise instructions. Reduces VRAM memory footprint and bandwidth by 8x.
 */

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
    const int lx = get_local_id(0); // 0 .. 31
    const int ly = get_local_id(1); // 0 .. 7
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    __local uchar s_tile[SH_H][SH_W];
    __local int l_changes;

    if (lx == 0 && ly == 0) {
        l_changes = 0;
    }

    // 1. Coalesced load of the core 32x8 tile
    s_tile[ly + 1][lx + 1] = (gy < height && gx < width) ? in_grid[gy * width + gx] : 1;

    // 2. Coalesced load of North halo (Warp 0: ly == 0)
    if (ly == 0) {
        const int north_gy = gy - 1;
        s_tile[0][lx + 1] = (north_gy >= 0 && gx < width) ? in_grid[north_gy * width + gx] : 1;
    }

    // 3. Coalesced load of South halo (Warp 7: ly == 7)
    if (ly == 7) {
        const int south_gy = gy + 1;
        s_tile[9][lx + 1] = (south_gy < height && gx < width) ? in_grid[south_gy * width + gx] : 1;
    }

    // 4. West and East boundary threads
    if (lx == 0) {
        const int west_gx = gx - 1;
        s_tile[ly + 1][0] = (gy < height && west_gx >= 0) ? in_grid[gy * width + west_gx] : 1;
    }
    if (lx == 31) {
        const int east_gx = gx + 1;
        s_tile[ly + 1][33] = (gy < height && east_gx < width) ? in_grid[gy * width + east_gx] : 1;
    }

    // 5. Four corner cells
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
            if (s_tile[ly][lx + 1] == 0) open_count++;     // North
            if (s_tile[ly + 2][lx + 1] == 0) open_count++; // South
            if (s_tile[ly + 1][lx] == 0) open_count++;     // West
            if (s_tile[ly + 1][lx + 2] == 0) open_count++; // East

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
    const int tid = ly * TILE_W + lx; // 0 .. 255

    __local uchar s_grid0[SUB_H][SUB_W];
    __local uchar s_grid1[SUB_H][SUB_W];
    __local int l_changes;

    if (tid == 0) {
        l_changes = 0;
    }

    // Origin of the R=2 halo in global coordinates
    const int base_x = get_group_id(0) * TILE_W - 2;
    const int base_y = get_group_id(1) * TILE_H - 2;

    // Load 36x12 = 432 bytes into s_grid0 using 256 threads
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

    // --- STEP 1: Compute inner 34x10 cells in SRAM ---
    const int TOTAL_STEP1 = (SUB_H - 2) * (SUB_W - 2); // 10 * 34 = 340 cells
    bool pruned_any = false;

    // First 256 cells
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

    // Remaining 84 cells (340 - 256)
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

    // --- STEP 2: Compute core 32x8 cells and write to global out_grid ---
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

    // If all 32 cells in this word are already walls (0xFFFFFFFF), nothing to prune!
    if (curr == 0xFFFFFFFFU) {
        out_words[idx] = 0xFFFFFFFFU;
        return;
    }

    // Fetch neighbor words
    const uint north = (r > 0) ? in_words[(r - 1) * words_per_row + col_word] : 0xFFFFFFFFU;
    const uint south = (r + 1 < height) ? in_words[(r + 1) * words_per_row + col_word] : 0xFFFFFFFFU;

    const uint west_word = (col_word > 0) ? in_words[r * words_per_row + (col_word - 1)] : 0xFFFFFFFFU;
    const uint east_word = (col_word + 1 < words_per_row) ? in_words[r * words_per_row + (col_word + 1)] : 0xFFFFFFFFU;

    // West neighbor is column c-1 (bit k-1 -> bit k: shift left by 1)
    // East neighbor is column c+1 (bit k+1 -> bit k: shift right by 1)
    const uint west = (curr << 1) | (west_word >> 31);
    const uint east = (curr >> 1) | (east_word << 31);

    // Inverted bitplanes: 1 = open PATH (0 in input)
    const uint w0 = ~west;
    const uint e0 = ~east;
    const uint n0 = ~north;
    const uint s0 = ~south;

    // 4-bitplane parallel full adder: count >= 2 in carry
    const uint sum1 = w0 ^ e0;
    const uint c1   = w0 & e0;

    const uint sum2 = sum1 ^ n0;
    const uint c2   = (sum1 & n0) | c1;

    const uint carry = (sum2 & s0) | c2; // bit is 1 iff open_neighbors >= 2

    // Prune mask: PATH cells (curr bit == 0) with open_neighbors <= 1 (carry bit == 0)
    uint prune_mask = (~curr) & (~carry);

    // Protect Start and End
    if (r == start_r && (start_c / 32) == col_word) {
        prune_mask &= ~(1U << (start_c % 32));
    }
    if (r == end_r && (end_c / 32) == col_word) {
        prune_mask &= ~(1U << (end_c % 32));
    }

    // Mask out of bounds bits on last word of row
    if (col_word == words_per_row - 1 && (width % 32) != 0) {
        const uint valid_bits = (1U << (width % 32)) - 1U;
        prune_mask &= valid_bits;
    }

    if (prune_mask != 0) {
        atomic_add(change_count, popcount(prune_mask));
    }

    out_words[idx] = curr | prune_mask;
}
