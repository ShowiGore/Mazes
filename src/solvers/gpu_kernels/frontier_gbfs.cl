/**
 * @file frontier_gbfs.cl
 * @brief GPU Bidirectional Beam-GBFS with Dual-Frontier Heuristic Bucketing (OpenCL)
 *
 * Implements branchless heuristic classification:
 * - primary frontier: active beam advancing towards target (Delta h <= 0)
 * - secondary queue: linear append buffer for backtracking (Delta h > 0)
 */

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
    __global int* f_pri_count,       // in/out
    __global int* f_sec_head,        // in/out
    __global int* f_sec_tail,        // in/out
    __global int* b_pri_count,       // in/out
    __global int* b_sec_head,        // in/out
    __global int* b_sec_tail,        // in/out
    __global int* state_grid,
    const int height,
    const int width,
    const int start_r,
    const int start_c,
    const int end_r,
    const int end_c,
    const int batch_limit,
    const int frontier_cap,
    __global int* collision_found,   // in/out
    __global int* collision_cell_A,  // out
    __global int* collision_cell_B,  // out
    __global int* steps_executed,    // in/out
    __global int* cells_visited_acc  // in/out
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

