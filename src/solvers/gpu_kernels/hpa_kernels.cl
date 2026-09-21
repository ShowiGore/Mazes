/**
 * @file hpa_kernels.cl
 * @brief GPU Hierarchical Pathfinding (HPA*) Kernels (OpenCL)
 *
 * Decomposes an H x W grid into 32x32 tiles.
 * Workgroups process tiles in parallel in __local SRAM to identify
 * border portals and compute intra-tile portal connectivity and distances.
 */

#define TILE_DIM 32

__kernel void hpa_find_horizontal_portals(
    __global const int* state_grid,
    const int height,
    const int width,
    const int num_tile_rows,
    const int num_tile_cols,
    __global int* portal_cell_A,
    __global int* portal_cell_B,
    __global int* portal_tile_A,
    __global int* portal_tile_B,
    __global int* portal_counter,
    const int max_portals
) {
    const int tc = get_global_id(0); // tile column
    const int tr = get_global_id(1); // tile row

    if (tr >= num_tile_rows - 1 || tc >= num_tile_cols) return;

    // Horizontal boundary between tile (tr, tc) and (tr + 1, tc)
    // Row in tile A is (tr + 1) * TILE_DIM - 1
    // Row in tile B is (tr + 1) * TILE_DIM
    const int r_A = (tr + 1) * TILE_DIM - 1;
    const int r_B = (tr + 1) * TILE_DIM;

    if (r_B >= height) return;

    const int c_start = tc * TILE_DIM;
    const int c_end = min((tc + 1) * TILE_DIM, width);

    for (int c = c_start; c < c_end; ++c) {
        const int idx_A = r_A * width + c;
        const int idx_B = r_B * width + c;

        // If both sides are PATH (not 0xFF wall)
        if (state_grid[idx_A] != 0xFF && state_grid[idx_B] != 0xFF) {
            const int pid = atomic_inc(portal_counter);
            if (pid < max_portals) {
                portal_cell_A[pid] = idx_A;
                portal_cell_B[pid] = idx_B;
                portal_tile_A[pid] = tr * num_tile_cols + tc;
                portal_tile_B[pid] = (tr + 1) * num_tile_cols + tc;
            }
        }
    }
}

__kernel void hpa_find_vertical_portals(
    __global const int* state_grid,
    const int height,
    const int width,
    const int num_tile_rows,
    const int num_tile_cols,
    __global int* portal_cell_A,
    __global int* portal_cell_B,
    __global int* portal_tile_A,
    __global int* portal_tile_B,
    __global int* portal_counter,
    const int max_portals
) {
    const int tc = get_global_id(0); // tile column
    const int tr = get_global_id(1); // tile row

    if (tr >= num_tile_rows || tc >= num_tile_cols - 1) return;

    // Vertical boundary between tile (tr, tc) and (tr, tc + 1)
    const int c_A = (tc + 1) * TILE_DIM - 1;
    const int c_B = (tc + 1) * TILE_DIM;

    if (c_B >= width) return;

    const int r_start = tr * TILE_DIM;
    const int r_end = min((tr + 1) * TILE_DIM, height);

    for (int r = r_start; r < r_end; ++r) {
        const int idx_A = r * width + c_A;
        const int idx_B = r * width + c_B;

        if (state_grid[idx_A] != 0xFF && state_grid[idx_B] != 0xFF) {
            const int pid = atomic_inc(portal_counter);
            if (pid < max_portals) {
                portal_cell_A[pid] = idx_A;
                portal_cell_B[pid] = idx_B;
                portal_tile_A[pid] = tr * num_tile_cols + tc;
                portal_tile_B[pid] = tr * num_tile_cols + (tc + 1);
            }
        }
    }
}

// Intra-tile path reconstruction: given a list of path segments (from_cell, to_cell)
// inside a tile, reconstruct the exact path in parallel
__kernel void hpa_reconstruct_tile_path(
    __global const int* state_grid,
    __global uchar* solution_grid,
    const int height,
    const int width,
    const int from_cell,
    const int to_cell
) {
    // Single-workgroup tile BFS to mark the path
    __local int local_parent[TILE_DIM * TILE_DIM];
    __local int local_queue[TILE_DIM * TILE_DIM];
    __local int q_head, q_tail, found;

    const int tid = get_local_id(0);
    const int lsize = get_local_size(0);

    for (int i = tid; i < TILE_DIM * TILE_DIM; i += lsize) {
        local_parent[i] = -1;
    }
    if (tid == 0) {
        q_head = 0;
        q_tail = 0;
        found = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int r_start = (from_cell / width / TILE_DIM) * TILE_DIM;
    const int c_start = (from_cell % width / TILE_DIM) * TILE_DIM;

    const int start_lr = (from_cell / width) - r_start;
    const int start_lc = (from_cell % width) - c_start;
    const int target_lr = (to_cell / width) - r_start;
    const int target_lc = (to_cell % width) - c_start;

    const int start_lidx = start_lr * TILE_DIM + start_lc;
    const int target_lidx = target_lr * TILE_DIM + target_lc;

    if (tid == 0) {
        local_parent[start_lidx] = start_lidx;
        local_queue[q_tail++] = start_lidx;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dr[4] = {-1, 0, 1, 0};
    const int dc[4] = {0, 1, 0, -1};

    while (q_head < q_tail && found == 0) {
        int curr = -1;
        if (tid == 0) {
            curr = local_queue[q_head++];
        }
        // Broadcast curr
        barrier(CLK_LOCAL_MEM_FENCE);
        curr = local_queue[q_head - 1];

        if (curr == target_lidx) {
            if (tid == 0) found = 1;
            break;
        }

        const int cr = curr / TILE_DIM;
        const int cc = curr % TILE_DIM;

        for (int d = 0; d < 4; ++d) {
            const int nr = cr + dr[d];
            const int nc = cc + dc[d];
            if (nr < 0 || nr >= TILE_DIM || nc < 0 || nc >= TILE_DIM) continue;
            const int gr = r_start + nr;
            const int gc = c_start + nc;
            if (gr >= height || gc >= width) continue;

            const int g_idx = gr * width + gc;
            if (state_grid[g_idx] == 0xFF) continue; // wall

            const int nl = nr * TILE_DIM + nc;
            if (local_parent[nl] == -1) {
                local_parent[nl] = curr;
                if (tid == 0) {
                    local_queue[q_tail++] = nl;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Trace back on thread 0
    if (tid == 0 && found) {
        int curr = target_lidx;
        while (curr != start_lidx) {
            const int cr = r_start + (curr / TILE_DIM);
            const int cc = c_start + (curr % TILE_DIM);
            solution_grid[cr * width + cc] = 1;
            curr = local_parent[curr];
        }
        const int cr = r_start + (start_lidx / TILE_DIM);
        const int cc = c_start + (start_lidx % TILE_DIM);
        solution_grid[cr * width + cc] = 1;
    }
}
