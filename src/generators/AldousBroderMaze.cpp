#include "AldousBroderMaze.hpp"
#include <vector>
#include <cstdint>

AldousBroderMaze::AldousBroderMaze(const int height, const int width, const unsigned int seed)
    : Maze(height, width, seed) {
    AldousBroderMaze::init();
    AldousBroderMaze::generate();
    buildStartEnd();
}

AldousBroderMaze::AldousBroderMaze(const int height, const int width)
    : Maze(height, width) {
    AldousBroderMaze::init();
    AldousBroderMaze::generate();
    buildStartEnd();
}

void AldousBroderMaze::init() {
    // Fill entire grid with walls
    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }
}

void AldousBroderMaze::generate() {
    // The maze graph nodes (cells) correspond to odd coordinates (2*r + 1, 2*c + 1)
    const int num_cell_rows = height / 2;
    const int num_cell_cols = width / 2;
    const int total_cells = num_cell_rows * num_cell_cols;

    if (total_cells <= 0) {
        return;
    }

    // Flat 1D visited array for optimal cache locality and O(1) lookups
    std::vector<uint8_t> visited(total_cells, 0);

    // Pick an initial cell uniformly at random
    int curr_r = randomInRange(0, num_cell_rows - 1);
    int curr_c = randomInRange(0, num_cell_cols - 1);

    visited[curr_r * num_cell_cols + curr_c] = 1;
    maze[2 * curr_r + 1][2 * curr_c + 1] = PATH;
    int visited_cells = 1;

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

    // Main Aldous-Broder random walk loop
    while (visited_cells < total_cells) {
        const int dir = next_random_dir();
        const int next_r = curr_r + d_row[dir];
        const int next_c = curr_c + d_col[dir];

        // Bounds check: if out of grid bounds, retry neighbor from current cell
        if (next_r < 0 || next_r >= num_cell_rows || next_c < 0 || next_c >= num_cell_cols) {
            continue;
        }

        const int next_idx = next_r * num_cell_cols + next_c;

        // If the neighbor has not been visited yet, carve a passage to it
        if (!visited[next_idx]) {
            visited[next_idx] = 1;

            // Carve intermediate wall between current cell and next cell:
            // Midpoint between (2*curr_r + 1) and (2*next_r + 1) is (curr_r + next_r + 1)
            maze[curr_r + next_r + 1][curr_c + next_c + 1] = PATH;

            // Carve the destination cell node
            maze[2 * next_r + 1][2 * next_c + 1] = PATH;

            ++visited_cells;
        }

        // Always step to the neighbor (standard random walk)
        curr_r = next_r;
        curr_c = next_c;
    }
}
