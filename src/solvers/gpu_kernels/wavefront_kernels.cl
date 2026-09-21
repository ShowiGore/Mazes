// =============================================================================
// WAVEFRONT & BITPACKED PRUNING KERNELS
// Ultra-scale bitpacked dead-end pruning + bidirectional wavefront expansion
// =============================================================================

// -----------------------------------------------------------------------------
// KERNEL 1: Bit-Packed 32-Cell SIMD Dead-End Pruning
// 1 = WALL (or pruned), 0 = OPEN PATH
// -----------------------------------------------------------------------------
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

    // If all 32 cells in this word are walls (0xFFFFFFFF), nothing to prune
    if (curr == 0xFFFFFFFFU) {
        out_words[idx] = 0xFFFFFFFFU;
        return;
    }

    const uint north = (r > 0) ? in_words[(r - 1) * words_per_row + col_word] : 0xFFFFFFFFU;
    const uint south = (r + 1 < height) ? in_words[(r + 1) * words_per_row + col_word] : 0xFFFFFFFFU;

    const uint west_word = (col_word > 0) ? in_words[r * words_per_row + (col_word - 1)] : 0xFFFFFFFFU;
    const uint east_word = (col_word + 1 < words_per_row) ? in_words[r * words_per_row + (col_word + 1)] : 0xFFFFFFFFU;

    // West neighbor is column c-1 (shift left)
    // East neighbor is column c+1 (shift right)
    const uint west = (curr << 1) | (west_word >> 31);
    const uint east = (curr >> 1) | (east_word << 31);

    // Inverted bitplanes: 1 = open PATH (0 in input)
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

    // Prune mask: PATH cells (curr == 0) with open_neighbors <= 1 (carry == 0)
    uint prune_mask = (~curr) & (~carry);

    // Protect Start and End
    if (r == start_r && (start_c / 32) == col_word) {
        prune_mask &= ~(1U << (start_c % 32));
    }
    if (r == end_r && (end_c / 32) == col_word) {
        prune_mask &= ~(1U << (end_c % 32));
    }

    // Mask out-of-bounds bits on last word of row
    if (col_word == words_per_row - 1 && (width % 32) != 0) {
        const uint valid_bits = (1U << (width % 32)) - 1U;
        prune_mask &= valid_bits;
    }

    if (prune_mask != 0) {
        atomic_add(change_count, popcount(prune_mask));
    }

    out_words[idx] = curr | prune_mask;
}

// -----------------------------------------------------------------------------
// KERNEL 2: Sparse Frontier Wavefront Expansion
// Expands active frontier cells into orthogonal unvisited path neighbors.
// Parent directions: 0 = North, 1 = South, 2 = West, 3 = East
// -----------------------------------------------------------------------------
__kernel void wavefront_expand_step(
    __global const uint* in_grid,           // Pruned maze: 1 = wall, 0 = path
    __global uint* visited_self,            // Bitpacked visited bitmap (1 = visited)
    __global const uint* visited_other,     // Bitpacked visited bitmap of the other search
    __global uint* parent_dir,              // 2 bits per cell parent directions
    __global const int2* curr_frontier,     // Array of active (row, col) cells
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
    if (tid >= curr_count) {
        return;
    }

    const int2 curr = curr_frontier[tid];
    const int r = curr.x;
    const int c = curr.y;

    // 4 directions: dr, dc, parent_dir_code (opposite direction back to curr)
    // 0: North (-1, 0) -> parent back to curr is South (1)
    // 1: South (+1, 0) -> parent back to curr is North (0)
    // 2: West  (0, -1) -> parent back to curr is East (3)
    // 3: East  (0, +1) -> parent back to curr is West (2)
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    const uint opp_dir[4] = {1U, 0U, 3U, 2U};

    for (int i = 0; i < 4; ++i) {
        const int nr = r + dr[i];
        const int nc = c + dc[i];

        if (nr < 0 || nr >= height || nc < 0 || nc >= width) {
            continue;
        }

        const int g_idx = nr * words_per_row + (nc / 32);
        const uint bit = 1U << (nc % 32);

        // Check if wall in pruned grid (1 = wall)
        if ((in_grid[g_idx] & bit) != 0) {
            continue;
        }

        // Atomic test-and-set on visited_self
        const uint prev_visited = atomic_or(&visited_self[g_idx], bit);
        if ((prev_visited & bit) != 0) {
            // Already visited by self in this or a prior step
            continue;
        }

        // Successfully claimed (nr, nc)! Record parent direction
        const int d_idx = nr * dir_words_per_row + (nc / 16);
        const uint d_shift = (nc % 16) * 2;
        atomic_or(&parent_dir[d_idx], (opp_dir[i] & 3U) << d_shift);

        // Check for collision with the other search's visited territory
        if ((visited_other[g_idx] & bit) != 0) {
            atomic_xchg(collision_flag, 1);
            *collision_cell_self = (int2)(nr, nc);
            *collision_cell_other = (int2)(nr, nc);
        }

        // Push to next frontier
        const int pos = atomic_inc(next_count);
        if (pos < max_frontier_size) {
            next_frontier[pos] = (int2)(nr, nc);
        }
    }
}

// -----------------------------------------------------------------------------
// KERNEL 3: In-VRAM Path Reconstruction
// Traces backwards from meet_cell to target_cell using 2-bit parent directions
// -----------------------------------------------------------------------------
__kernel void trace_path(
    __global const uint* parent_dir,
    const int2 target_cell,
    const int2 meet_cell,
    __global int2* out_path,
    __global int* out_length,
    const int max_path_cells,
    const int dir_words_per_row
) {
    if (get_global_id(0) != 0) {
        return;
    }

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

        if (dir == 0)      r -= 1; // North
        else if (dir == 1) r += 1; // South
        else if (dir == 2) c -= 1; // West
        else if (dir == 3) c += 1; // East

        if (len < max_path_cells) {
            out_path[len++] = (int2)(r, c);
        } else {
            break; // Safety cap
        }
    }

    *out_length = len;
}
