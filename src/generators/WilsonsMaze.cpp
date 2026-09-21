#include "WilsonsMaze.hpp"
#include <vector>
#include <cstdint>

WilsonsMaze::WilsonsMaze(const int height, const int width, const unsigned int seed)
    : Maze(height, width, seed) {
    this->generator_name = "wilson";
    WilsonsMaze::init();
    WilsonsMaze::generate();
    buildStartEnd();
}

WilsonsMaze::WilsonsMaze(const int height, const int width)
    : Maze(height, width) {
    this->generator_name = "wilson";
    WilsonsMaze::init();
    WilsonsMaze::generate();
    buildStartEnd();
}

void WilsonsMaze::init() {
    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }
}

void WilsonsMaze::generate() {
    const size_t num_cell_rows = static_cast<size_t>(height) / 2;
    const size_t num_cell_cols = static_cast<size_t>(width) / 2;
    const size_t total_cells = num_cell_rows * num_cell_cols;

    if (total_cells == 0) {
        return;
    }

    std::vector<uint8_t> in_tree(total_cells, 0);
    std::vector<uint8_t> next_dir(total_cells, 0);

    constexpr int d_row[4] = {-1, 0, 1, 0};
    constexpr int d_col[4] = {0, 1, 0, -1};

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
    const int root_r = randomInRange(0, static_cast<int>(num_cell_rows - 1));
    const int root_c = randomInRange(0, static_cast<int>(num_cell_cols - 1));
    const size_t root_idx = static_cast<size_t>(root_r) * num_cell_cols + root_c;

    in_tree[root_idx] = 1;
    maze[2 * root_r + 1][2 * root_c + 1] = PATH;

    // Step 2: Iterate through all cells
    for (size_t cell_idx = 0; cell_idx < total_cells; ++cell_idx) {
        if (in_tree[cell_idx]) {
            continue;
        }

        // --- Phase 1: Loop-Erased Random Walk (LERW) ---
        size_t curr_idx = cell_idx;
        int curr_r = static_cast<int>(curr_idx / num_cell_cols);
        int curr_c = static_cast<int>(curr_idx % num_cell_cols);

        while (!in_tree[curr_idx]) {
            const int dir = next_random_dir();
            const int next_r = curr_r + d_row[dir];
            const int next_c = curr_c + d_col[dir];

            if (next_r < 0 || static_cast<size_t>(next_r) >= num_cell_rows ||
                next_c < 0 || static_cast<size_t>(next_c) >= num_cell_cols) {
                continue;
            }

            next_dir[curr_idx] = static_cast<uint8_t>(dir);

            curr_r = next_r;
            curr_c = next_c;
            curr_idx = static_cast<size_t>(next_r) * num_cell_cols + next_c;
        }

        // --- Phase 2: Retrace and Carve ---
        size_t path_idx = cell_idx;
        int path_r = static_cast<int>(path_idx / num_cell_cols);
        int path_c = static_cast<int>(path_idx % num_cell_cols);

        while (!in_tree[path_idx]) {
            const int dir = next_dir[path_idx];
            const int next_r = path_r + d_row[dir];
            const int next_c = path_c + d_col[dir];

            in_tree[path_idx] = 1;
            maze[2 * path_r + 1][2 * path_c + 1] = PATH;
            maze[path_r + next_r + 1][path_c + next_c + 1] = PATH;

            path_r = next_r;
            path_c = next_c;
            path_idx = static_cast<size_t>(next_r) * num_cell_cols + next_c;
        }
    }
}
