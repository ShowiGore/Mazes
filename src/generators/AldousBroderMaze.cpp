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
    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }
}

void AldousBroderMaze::generate() {
    const size_t num_cell_rows = static_cast<size_t>(height) / 2;
    const size_t num_cell_cols = static_cast<size_t>(width) / 2;
    const size_t total_cells = num_cell_rows * num_cell_cols;

    if (total_cells == 0) {
        return;
    }

    std::vector<uint8_t> visited(total_cells, 0);

    int curr_r = randomInRange(0, static_cast<int>(num_cell_rows - 1));
    int curr_c = randomInRange(0, static_cast<int>(num_cell_cols - 1));

    visited[static_cast<size_t>(curr_r) * num_cell_cols + curr_c] = 1;
    maze[2 * curr_r + 1][2 * curr_c + 1] = PATH;
    size_t visited_cells = 1;

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

    while (visited_cells < total_cells) {
        const int dir = next_random_dir();
        const int next_r = curr_r + d_row[dir];
        const int next_c = curr_c + d_col[dir];

        if (next_r < 0 || static_cast<size_t>(next_r) >= num_cell_rows ||
            next_c < 0 || static_cast<size_t>(next_c) >= num_cell_cols) {
            continue;
        }

        const size_t next_idx = static_cast<size_t>(next_r) * num_cell_cols + next_c;

        if (!visited[next_idx]) {
            visited[next_idx] = 1;
            maze[curr_r + next_r + 1][curr_c + next_c + 1] = PATH;
            maze[2 * next_r + 1][2 * next_c + 1] = PATH;
            ++visited_cells;
        }

        curr_r = next_r;
        curr_c = next_c;
    }
}
