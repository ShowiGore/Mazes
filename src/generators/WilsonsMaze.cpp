#include "WilsonsMaze.hpp"
#include <algorithm>

WilsonsMaze::WilsonsMaze(const int height, const int width, const unsigned int seed) : Maze(height, width, seed) {
    init();
    WilsonsMaze::generate();
    buildStartEnd();
}

WilsonsMaze::WilsonsMaze(const int height, const int width) : Maze(height, width) {
    init();
    WilsonsMaze::generate();
    buildStartEnd();
}

void WilsonsMaze::init () {

    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }

    walk_grid.assign(height / 2, std::vector<WalkDirections>(width / 2, UP));

}

void WilsonsMaze::generate() {

    std::vector<std::pair<int, int>> cells; //valid cell coordinates
    cells.reserve((height / 2) * (width / 2));
    for (int h = 1; h < height; h += 2) {
        for (int w = 1; w < width; w += 2) {
            cells.emplace_back(h, w);
        }
    }
    std::ranges::shuffle(cells, this->re); // Shuffle the list to ensure uniform probability of selection.

    if (!cells.empty()) { // Pick the first cell and mark it as part of the maze
        auto [h, w] = cells.back();
        cells.pop_back();
        maze[h][w] = PATH;
    }

    // Iterate through the shuffled list
    for (const auto& [h, w] : cells) {
        if (maze[h][w] == WALL) {
            loopErasedRandomWalk(h, w);
        } // If a cell is already part of the maze skip it
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

        std::vector<WalkDirections> valid_dirs;
        valid_dirs.reserve(4);

        for (int i = 0; i < 4; ++i) {
            // Check neighbors using the cell step (2)
            const int nr = current_row + d_row_cell[i];
            const int nc = current_column + d_col_cell[i];

            if (nr > 0 && nr < height - 1 && nc > 0 && nc < width - 1) {
                valid_dirs.push_back(static_cast<WalkDirections>(i));
            }
        }

        if (!valid_dirs.empty()) {
            const int rand_index = randomInRange(0, valid_dirs.size() - 1);
            const WalkDirections dir = valid_dirs[rand_index];

            // Store direction in the reduced grid
            walk_grid[current_row / 2][current_column / 2] = dir;

            // Move to next cell
            current_row += d_row_cell[dir];
            current_column += d_col_cell[dir];
        }
    }

    //Retrace and carve
    current_row = start_row;
    current_column = start_column;

    while (maze[current_row][current_column] == WALL) {

        // Used to find the wall between nodes (distance 1)
        constexpr int d_col_wall[] = {0, 1, 0, -1};
        constexpr int d_row_wall[] = {-1, 0, 1, 0};

        const WalkDirections dir = walk_grid[current_row / 2][current_column / 2];

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

