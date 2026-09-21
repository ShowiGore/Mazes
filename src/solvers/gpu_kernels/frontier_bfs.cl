/**
 * @file frontier_bfs.cl
 * @brief GPU Parallel Frontier Expansion for Bidirectional BFS Solver (Native 32-bit atomics)
 */

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

    // 4 orthogonal directions: UP, RIGHT, DOWN, LEFT
    const int dr[4] = {-1, 0, 1, 0};
    const int dc[4] = {0, 1, 0, -1};
    // Parent direction from neighbor back to curr
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
            continue; // Wall
        }

        // Check if opposing frontier was reached
        if ((n_val & 0xF0) == opp_tag) {
            *collision_cell_A = curr_idx;
            *collision_cell_B = n_idx;
            atomic_xchg(collision_found, 1);
            return;
        }

        if (n_val == 0) {
            // Unvisited PATH cell - claim it with 32-bit native atomic CAS
            const int new_val = search_tag | parent_dir[i];
            const int old = atomic_cmpxchg(&state_grid[n_idx], 0, new_val);

            if (old == 0) {
                // Successfully claimed! Add to next frontier
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

// =============================================================================
// KERNEL 2: Multi-Step Persistent Batched Frontier Expansion
// =============================================================================
__kernel void expand_frontier_batched(
    __global int* f_curr,
    __global int* f_next,
    __global int* b_curr,
    __global int* b_next,
    __global int* f_count_buf,       // in/out
    __global int* b_count_buf,       // in/out
    __global int* state_grid,
    const int height,
    const int width,
    const int batch_limit,
    __global int* collision_found,   // in/out
    __global int* collision_cell_A,  // out
    __global int* collision_cell_B,  // out
    __global int* steps_executed     // in/out
) {
    __local int l_f_curr_count;
    __local int l_f_next_count;
    __local int l_b_curr_count;
    __local int l_b_next_count;
    __local int l_collision;
    __local int l_step;
    __local int l_f_curr_is_buf0;
    __local int l_b_curr_is_buf0;

    const int tid = get_local_id(0);
    const int lsize = get_local_size(0);

    if (tid == 0) {
        l_f_curr_count = *f_count_buf;
        l_f_next_count = 0;
        l_b_curr_count = *b_count_buf;
        l_b_next_count = 0;
        l_collision = *collision_found;
        l_step = 0;
        l_f_curr_is_buf0 = 1;
        l_b_curr_is_buf0 = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int dr[4] = {-1, 0, 1, 0};
    const int dc[4] = {0, 1, 0, -1};
    const int pdir[4] = {DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT};

    while (l_step < batch_limit && l_collision == 0 && l_f_curr_count > 0 && l_b_curr_count > 0) {
        __global int* f_in  = l_f_curr_is_buf0 ? f_curr : f_next;
        __global int* f_out = l_f_curr_is_buf0 ? f_next : f_curr;
        __global int* b_in  = l_b_curr_is_buf0 ? b_curr : b_next;
        __global int* b_out = l_b_curr_is_buf0 ? b_next : b_curr;

        // --- 1. Forward Expansion ---
        for (int i = tid; i < l_f_curr_count; i += lsize) {
            if (l_collision != 0) break;
            const int curr_idx = f_in[i];
            const int r = curr_idx / width;
            const int c = curr_idx % width;

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
                        const int pos = atomic_inc(&l_f_next_count);
                        f_out[pos] = n_idx;
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
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_collision != 0) break;

        // --- 2. Backward Expansion ---
        for (int i = tid; i < l_b_curr_count; i += lsize) {
            if (l_collision != 0) break;
            const int curr_idx = b_in[i];
            const int r = curr_idx / width;
            const int c = curr_idx % width;

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
                        const int pos = atomic_inc(&l_b_next_count);
                        b_out[pos] = n_idx;
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
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l_collision != 0) break;

        // --- 3. Advance to next step ---
        if (tid == 0) {
            l_step++;
            l_f_curr_count = l_f_next_count;
            l_f_next_count = 0;
            l_b_curr_count = l_b_next_count;
            l_b_next_count = 0;
            l_f_curr_is_buf0 = !l_f_curr_is_buf0;
            l_b_curr_is_buf0 = !l_b_curr_is_buf0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Ensure f_curr and b_curr contain the current frontier upon exit
    if (l_f_curr_is_buf0 == 0) {
        for (int i = tid; i < l_f_curr_count; i += lsize) {
            f_curr[i] = f_next[i];
        }
    }
    if (l_b_curr_is_buf0 == 0) {
        for (int i = tid; i < l_b_curr_count; i += lsize) {
            b_curr[i] = b_next[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0) {
        *f_count_buf = l_f_curr_count;
        *b_count_buf = l_b_curr_count;
        atomic_add(steps_executed, l_step);
    }
}

