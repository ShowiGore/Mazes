#include "WilsonsMaze.hpp"
#include <algorithm>

WilsonsMaze::WilsonsMaze(const int height, const int width, const unsigned int seed) : Maze(height, width, seed) {
    WilsonsMaze::init();
    WilsonsMaze::generate();
    buildStartEnd();
}

WilsonsMaze::WilsonsMaze(const int height, const int width) : Maze(height, width) {
    WilsonsMaze::init();
    WilsonsMaze::generate();
    buildStartEnd();
}

void WilsonsMaze::init () {

    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }

    walk_grid.assign(height / 2, std::vector<Direction>(width / 2, UP));

}

void WilsonsMaze::generate() {

    // Gen valid cell coordinates
    std::vector<std::pair<int, int>> cells;
    cells.reserve((height / 2) * (width / 2));
    for (int h = 1; h < height; h += 2) {
        for (int w = 1; w < width; w += 2) {
            cells.emplace_back(h, w);
        }
    }
    std::ranges::shuffle(cells, this->re);

    // Gen first path from center to edge
    if (!cells.empty()) {
        // Force odd index to ensure a valid cell
        const int center_h = (height / 2) | 1;
        const int center_w = (width / 2) | 1;
        loop_erased_random_walk_initial(center_h, center_w);
    }

    // Iterate through the shuffled list
    for (const auto& [h, w] : cells) {
        if (maze[h][w] == WALL) {
            loopErasedRandomWalk(h, w);
        }
    }
}

void WilsonsMaze::loopErasedRandomWalk(const int start_row, const int start_column) {
    int current_row = start_row;
    int current_column = start_column;

    // Used to jump between nodes (distance 2)
    constexpr int d_row_cell[] = {-2, 0, 2, 0};
    constexpr int d_col_cell[] = {0, 2, 0, -2};

    // Random Walk
    while (maze[current_row][current_column] == WALL) {

        std::vector<Direction> valid_dirs;
        valid_dirs.reserve(4);

        for (int i = 0; i < 4; ++i) {
            // Check neighbors using the cell step (2)
            const int nr = current_row + d_row_cell[i];
            const int nc = current_column + d_col_cell[i];

            if (nr > 0 && nr < height - 1 && nc > 0 && nc < width - 1) {
                valid_dirs.push_back(static_cast<Direction>(i));
            }
        }

        if (!valid_dirs.empty()) {
            const int rand_index = randomInRange(0, valid_dirs.size() - 1);
            const Direction dir = valid_dirs[rand_index];

            // Store direction in the reduced grid
            walk_grid[current_row / 2][current_column / 2] = dir;

            // Move to next cell
            current_row += d_row_cell[dir];
            current_column += d_col_cell[dir];
        }
    }

    // Retrace and carve
    current_row = start_row;
    current_column = start_column;

    while (maze[current_row][current_column] == WALL) {

        // Used to find the wall between nodes (distance 1)
        constexpr int d_col_wall[] = {0, 1, 0, -1};
        constexpr int d_row_wall[] = {-1, 0, 1, 0};

        const Direction dir = walk_grid[current_row / 2][current_column / 2];

        // Mark current node
        maze[current_row][current_column] = PATH;

        // Carve intermediate wall
        const int wall_r = current_row + d_row_wall[dir];
        const int wall_c = current_column + d_col_wall[dir];
        maze[wall_r][wall_c] = PATH;

        // Move to next node
        current_row += d_row_cell[dir];
        current_column += d_col_cell[dir];
    }
}

void WilsonsMaze::loop_erased_random_walk_initial(const int start_row, const int start_column) {
    int current_row = start_row;
    int current_column = start_column;

    constexpr int d_row_cell[] = {-2, 0, 2, 0};
    constexpr int d_col_cell[] = {0, 2, 0, -2};

    // Random Walk from random cell to edge
    auto is_at_border = [&](const int r, const int c) {
        return (r == 1 || r == height - 2 || c == 1 || c == width - 2);
    };

    // If first cell is already at the edge, the loop does not execute
    while (!is_at_border(current_row, current_column)) {

        std::vector<Direction> valid_dirs;
        valid_dirs.reserve(4);

        for (int i = 0; i < 4; ++i) {
            const int nr = current_row + d_row_cell[i];
            const int nc = current_column + d_col_cell[i];

            // Move only inside valid bounds
            if (nr > 0 && nr < height - 1 && nc > 0 && nc < width - 1) {
                valid_dirs.push_back(static_cast<Direction>(i));
            }
        }

        if (!valid_dirs.empty()) {
            const int rand_index = randomInRange(0, valid_dirs.size() - 1);
            const Direction dir = valid_dirs[rand_index];

            // Save direction
            walk_grid[current_row / 2][current_column / 2] = dir;

            current_row += d_row_cell[dir];
            current_column += d_col_cell[dir];
        }
    }

    // Retrace and carve
    current_row = start_row;
    current_column = start_column;

    maze[current_row][current_column] = PATH;

    // Reset pointers to start to trace path
    int trace_row = start_row;
    int trace_col = start_column;

    // Until arrival to objective cell
    while (maze[trace_row][trace_col] == WALL) {

        constexpr int d_col_wall[] = {0, 1, 0, -1};
        constexpr int d_row_wall[] = {-1, 0, 1, 0};

        const Direction dir = walk_grid[trace_row / 2][trace_col / 2];

        maze[trace_row][trace_col] = PATH;

        const int wall_r = trace_row + d_row_wall[dir];
        const int wall_c = trace_col + d_col_wall[dir];
        maze[wall_r][wall_c] = PATH;

        trace_row += d_row_cell[dir];
        trace_col += d_col_cell[dir];
    }
}
