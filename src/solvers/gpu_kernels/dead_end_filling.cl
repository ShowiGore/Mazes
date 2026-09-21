/**
 * @file dead_end_filling.cl
 * @brief OpenCL Cellular Automaton Kernel for Parallel Dead-End Filling
 *
 * In each iteration, every open PATH cell (value 0) that has <= 1 open orthogonal
 * neighbors (and is not the protected Start or End cell) is pruned into a WALL (value 1).
 *
 * An atomic reduction tracks the total number of pruned cells in each iteration.
 * The solver converges when change_count == 0.
 */

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

    // Work-group local accumulator to avoid global VRAM atomic contention
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
            // Already a wall (1), stays a wall
            out_grid[idx] = 1;
        } else if ((r == start_r && c == start_c) || (r == end_r && c == end_c)) {
            // Start and End cells are protected and never pruned
            out_grid[idx] = 0;
        } else {
            // Count open (PATH == 0) orthogonal neighbors
            int open_count = 0;
            if (r > 0 && in_grid[(r - 1) * width + c] == 0) open_count++;
            if (r < height - 1 && in_grid[(r + 1) * width + c] == 0) open_count++;
            if (c > 0 && in_grid[r * width + (c - 1)] == 0) open_count++;
            if (c < width - 1 && in_grid[r * width + (c + 1)] == 0) open_count++;

            if (open_count <= 1) {
                // Prune dead-end cell into a wall
                out_grid[idx] = 1;
                pruned = true;
            } else {
                // Cell remains an open path
                out_grid[idx] = 0;
            }
        }
    }

    if (pruned) {
        atomic_inc(&l_changes);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Single atomic add to global memory per work-group if changes occurred
    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        if (l_changes > 0) {
            atomic_add(change_count, l_changes);
        }
    }
}
