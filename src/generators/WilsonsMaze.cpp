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

}

void WilsonsMaze::generate() {

    std::vector<std::pair<int, int>> cells((height / 2) * (width / 2)); //valid cell coordinates
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

void WilsonsMaze::loopErasedRandomWalk(int startR, int startC) {
    int currR = startR;
    int currC = startC;

    // Deltas for UP, RIGHT, DOWN, LEFT
    const int dr[] = {-1, 0, 1, 0}; // Standard step (1 unit) - wait, grid logic needs step of 2?
    // In grid mazes, neighbors are usually 2 units away (jumping over the wall).
    // But strict Wilson's walks on the graph nodes.
    // Let's assume we move 2 units in the array to reach the next "cell".
    const int step = 2;
    const int dRow[] = {-step, 0, step, 0};
    const int dCol[] = {0, step, 0, -step};

    // --- Phase 1: The Random Walk ---
    // Walk until we hit a cell that is already part of the maze (PATH).
    while (maze[currR][currC] == WALL) {

        // Pick a random valid direction
        std::vector<int> validDirs;
        for (int i = 0; i < 4; ++i) {
            int nr = currR + dRow[i];
            int nc = currC + dCol[i];
            // Check bounds (assuming valid cells are within [1, height-2])
            if (nr > 0 && nr < height - 1 && nc > 0 && nc < width - 1) {
                validDirs.push_back(i);
            }
        }

        if (!validDirs.empty()) {
            // Pick random direction
            int randIndex = randomInRange(0, validDirs.size() - 1);
            int dir = validDirs[randIndex];

            // **Crucial**: We store the direction at the current cell.
            // If we revisit this cell later in the same walk (creating a loop),
            // this value gets overwritten, effectively cutting off the loop.
            walkGrid[currR][currC] = dir;

            // Move to next cell
            currR += dRow[dir];
            currC += dCol[dir];
        }
    }

    // --- Phase 2: Retrace and Carve ---
    // We reached the maze. Now restart from the beginning and follow the arrows.
    currR = startR;
    currC = startC;

    while (maze[currR][currC] == WALL) {
        int dir = walkGrid[currR][currC];

        // 1. Mark current cell as PATH
        maze[currR][currC] = PATH;

        // 2. Carve the wall between current and next
        // The wall is at current + (delta / 2)
        int wallR = currR + (dRow[dir] / 2);
        int wallC = currC + (dCol[dir] / 2);
        maze[wallR][wallC] = PATH;

        // 3. Move to next cell
        currR += dRow[dir];
        currC += dCol[dir];
    }
}

