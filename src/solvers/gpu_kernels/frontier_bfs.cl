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
