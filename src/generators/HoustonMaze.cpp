#include "HoustonMaze.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>

// Default constructor using the theoretical optimal threshold alpha = 1/3
HoustonMaze::HoustonMaze(const int height, const int width, const unsigned int seed)
    : HoustonMaze(height, width, seed, THEORETICAL_ALPHA) {}

HoustonMaze::HoustonMaze(const int height, const int width)
    : HoustonMaze(height, width, THEORETICAL_ALPHA) {}

// Custom threshold constructor
HoustonMaze::HoustonMaze(const int height, const int width, const unsigned int seed, const float switch_threshold)
    : Maze(height, width, seed), switch_threshold(std::clamp(switch_threshold, 0.01f, 0.99f)) {
    HoustonMaze::init();
    HoustonMaze::generate();
    buildStartEnd();
}

HoustonMaze::HoustonMaze(const int height, const int width, const float switch_threshold)
    : Maze(height, width), switch_threshold(std::clamp(switch_threshold, 0.01f, 0.99f)) {
    HoustonMaze::init();
    HoustonMaze::generate();
    buildStartEnd();
}

void HoustonMaze::init() {
    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }
}

void HoustonMaze::generate() {
    const int num_cell_rows = height / 2;
    const int num_cell_cols = width / 2;
    const int total_cells = num_cell_rows * num_cell_cols;

    if (total_cells <= 0) {
        return;
    }

    // in_tree[idx]: 1 if cell is part of the spanning tree, 0 otherwise
    std::vector<uint8_t> in_tree(total_cells, 0);

    // next_dir[idx]: direction buffer for Wilson's phase loop-erased random walk
    std::vector<uint8_t> next_dir(total_cells, 0);

    // Movement deltas matching Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3
    constexpr int d_row[4] = {-1, 0, 1, 0};
    constexpr int d_col[4] = {0, 1, 0, -1};

    // Bit-buffered PRNG helper (16 directions per 32-bit PRNG call)
    uint32_t rng_buffer = 0;
    int bits_remaining = 0;

    auto next_random_dir = [&]() -> int {
        if (bits_remaining < 2) {
            rng_buffer = this->re();
            bits_remaining = 32;
        }
        const int dir = rng_buffer & 3;
        rng_buffer >>= 2;
        bits_remaining -= 2;
        return dir;
    };

    // Calculate threshold cell count for switching from Aldous-Broder to Wilson.
    // If using the theoretical alpha (1/3), use exact integer division; otherwise use float math.
    const int threshold_cells = (switch_threshold == THEORETICAL_ALPHA)
        ? std::max(1, total_cells / 3)
        : std::max(1, static_cast<int>(total_cells * switch_threshold));

    // =========================================================================
    // PHASE 1: Aldous-Broder Algorithm (Early Game Speed)
    // Run simple random walk until threshold_cells have been added to the tree.
    // =========================================================================
    int curr_r = randomInRange(0, num_cell_rows - 1);
    int curr_c = randomInRange(0, num_cell_cols - 1);

    in_tree[curr_r * num_cell_cols + curr_c] = 1;
    maze[2 * curr_r + 1][2 * curr_c + 1] = PATH;
    int visited_cells = 1;

    while (visited_cells < threshold_cells) {
        const int dir = next_random_dir();
        const int next_r = curr_r + d_row[dir];
        const int next_c = curr_c + d_col[dir];

        if (next_r < 0 || next_r >= num_cell_rows || next_c < 0 || next_c >= num_cell_cols) {
            continue;
        }

        const int next_idx = next_r * num_cell_cols + next_c;
        if (!in_tree[next_idx]) {
            in_tree[next_idx] = 1;

            // Carve intermediate wall and target cell
            maze[curr_r + next_r + 1][curr_c + next_c + 1] = PATH;
            maze[2 * next_r + 1][2 * next_c + 1] = PATH;

            ++visited_cells;
        }

        curr_r = next_r;
        curr_c = next_c;
    }

    // =========================================================================
    // PHASE 2: Wilson's Algorithm (Late Game Speed)
    // Seamlessly transition: the tree T_0 built by Aldous-Broder acts as the
    // initial target set. We use Loop-Erased Random Walks (LERW) for remaining cells.
    // =========================================================================
    for (int cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        if (in_tree[cell_idx]) {
            continue;
        }

        // --- Step 2A: Loop-Erased Random Walk ---
        int walk_idx = cell_idx;
        int walk_r = walk_idx / num_cell_cols;
        int walk_c = walk_idx % num_cell_cols;

        while (!in_tree[walk_idx]) {
            const int dir = next_random_dir();
            const int next_r = walk_r + d_row[dir];
            const int next_c = walk_c + d_col[dir];

            if (next_r < 0 || next_r >= num_cell_rows || next_c < 0 || next_c >= num_cell_cols) {
                continue;
            }

            // Overwriting next_dir automatically erases loops in O(1)
            next_dir[walk_idx] = static_cast<uint8_t>(dir);

            walk_r = next_r;
            walk_c = next_c;
            walk_idx = next_r * num_cell_cols + next_c;
        }

        // --- Step 2B: Retrace and Carve into the Tree ---
        int path_idx = cell_idx;
        int path_r = path_idx / num_cell_cols;
        int path_c = path_idx % num_cell_cols;

        while (!in_tree[path_idx]) {
            const int dir = next_dir[path_idx];
            const int next_r = path_r + d_row[dir];
            const int next_c = path_c + d_col[dir];

            in_tree[path_idx] = 1;
            maze[2 * path_r + 1][2 * path_c + 1] = PATH;
            maze[path_r + next_r + 1][path_c + next_c + 1] = PATH;

            path_r = next_r;
            path_c = next_c;
            path_idx = next_r * num_cell_cols + next_c;
        }
    }
}
