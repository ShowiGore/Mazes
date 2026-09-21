#include "WilsonsMaze.hpp"
#include <vector>
#include <cstdint>

WilsonsMaze::WilsonsMaze(const int height, const int width, const unsigned int seed)
    : Maze(height, width, seed) {
    WilsonsMaze::init();
    WilsonsMaze::generate();
    buildStartEnd();
}

WilsonsMaze::WilsonsMaze(const int height, const int width)
    : Maze(height, width) {
    WilsonsMaze::init();
    WilsonsMaze::generate();
    buildStartEnd();
}

void WilsonsMaze::init() {
    // Fill entire grid with walls
    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }
}

void WilsonsMaze::generate() {
    // The maze graph nodes (cells) correspond to odd coordinates (2*r + 1, 2*c + 1)
    const int num_cell_rows = height / 2;
    const int num_cell_cols = width / 2;
    const int total_cells = num_cell_rows * num_cell_cols;

    if (total_cells <= 0) {
        return;
    }

    // in_tree[idx]: 1 if cell is part of the spanning tree, 0 otherwise
    std::vector<uint8_t> in_tree(total_cells, 0);

    // next_dir[idx]: records the direction (0..3) taken from cell during the random walk.
    // Overwriting this value automatically performs loop erasure in O(1) time.
    std::vector<uint8_t> next_dir(total_cells, 0);

    // Movement deltas matching the Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3
    constexpr int d_row[4] = {-1, 0, 1, 0};
    constexpr int d_col[4] = {0, 1, 0, -1};

    // Bit-buffered PRNG helper:
    // A single 32-bit random number from mt19937 provides 16 two-bit directions (0 to 3).
    // This reduces PRNG invocations by 93.75% with zero modulo bias.
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

    // Step 1: Pick an initial cell uniformly at random to root the tree
    const int root_r = randomInRange(0, num_cell_rows - 1);
    const int root_c = randomInRange(0, num_cell_cols - 1);
    const int root_idx = root_r * num_cell_cols + root_c;

    in_tree[root_idx] = 1;
    maze[2 * root_r + 1][2 * root_c + 1] = PATH;

    // Step 2: Iterate through all cells.
    // Wilson's theorem (1996) proves that the resulting spanning tree distribution
    // is independent of the order in which walk roots are chosen. Sequential scanning
    // provides optimal cache locality and avoids allocating/shuffling coordinates.
    for (int cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        if (in_tree[cell_idx]) {
            continue;
        }

        // --- Phase 1: Loop-Erased Random Walk (LERW) ---
        // Walk until we hit any cell that is already part of the tree.
        int curr_idx = cell_idx;
        int curr_r = curr_idx / num_cell_cols;
        int curr_c = curr_idx % num_cell_cols;

        while (!in_tree[curr_idx]) {
            const int dir = next_random_dir();
            const int next_r = curr_r + d_row[dir];
            const int next_c = curr_c + d_col[dir];

            // Bounds check
            if (next_r < 0 || next_r >= num_cell_rows || next_c < 0 || next_c >= num_cell_cols) {
                continue;
            }

            // Record direction taken. If we revisit curr_idx, the previous direction is
            // overwritten, which inherently erases any loops without searching!
            next_dir[curr_idx] = static_cast<uint8_t>(dir);

            curr_r = next_r;
            curr_c = next_c;
            curr_idx = next_r * num_cell_cols + next_c;
        }

        // --- Phase 2: Retrace and Carve ---
        // Retrace the loop-erased path from cell_idx to the tree, carving passages.
        int path_idx = cell_idx;
        int path_r = path_idx / num_cell_cols;
        int path_c = path_idx % num_cell_cols;

        while (!in_tree[path_idx]) {
            const int dir = next_dir[path_idx];
            const int next_r = path_r + d_row[dir];
            const int next_c = path_c + d_col[dir];

            // Mark current cell as part of the tree and carve it
            in_tree[path_idx] = 1;
            maze[2 * path_r + 1][2 * path_c + 1] = PATH;

            // Carve intermediate wall between current cell and next cell
            maze[path_r + next_r + 1][path_c + next_c + 1] = PATH;

            path_r = next_r;
            path_c = next_c;
            path_idx = next_r * num_cell_cols + next_c;
        }
    }
}
