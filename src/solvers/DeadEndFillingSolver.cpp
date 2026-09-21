#include "DeadEndFillingSolver.hpp"
#include <vector>
#include <cstdint>

bool DeadEndFillingSolver::solve(const Maze &maze_object) {
    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();

    this->visited.assign(this->height, std::vector<bool>(this->width, false));
    this->solution.assign(this->height, std::vector<bool>(this->width, false));

    const size_t total_grid_cells = static_cast<size_t>(this->height) * this->width;

    // Flat 1D degree array: stores count of adjacent PATH neighbors.
    // Sentinel value 0xFF represents a cell that has been pruned as a dead end.
    // This single array eliminates the need for a separate 'pruned' array, saving ~17 GB on 131073x131073!
    constexpr uint8_t PRUNED = 0xFF;
    std::vector<uint8_t> degree(total_grid_cells, 0);

    std::vector<size_t> dead_end_queue;
    dead_end_queue.reserve(total_grid_cells / 8);

    constexpr int d_row[4] = {-1, 0, 1, 0};
    constexpr int d_col[4] = {0, 1, 0, -1};

    // Pass 1: Compute degree for all PATH cells and collect initial dead ends
    for (int r = 0; r < this->height; ++r) {
        for (int c = 0; c < this->width; ++c) {
            if (grid[r][c] == WALL) {
                continue;
            }

            int open_neighbors = 0;
            for (int i = 0; i < 4; ++i) {
                const int nr = r + d_row[i];
                const int nc = c + d_col[i];
                if (nr >= 0 && nr < this->height && nc >= 0 && nc < this->width && grid[nr][nc] == PATH) {
                    ++open_neighbors;
                }
            }

            const size_t idx = static_cast<size_t>(r) * this->width + c;
            degree[idx] = static_cast<uint8_t>(open_neighbors);

            const bool is_start_or_end = (r == this->start.first && c == this->start.second) ||
                                         (r == this->end.first && c == this->end.second);

            if (!is_start_or_end && open_neighbors <= 1) {
                degree[idx] = PRUNED;
                dead_end_queue.push_back(idx);
            }
        }
    }

    // Pass 2: Iteratively prune dead ends
    size_t head = 0;
    while (head < dead_end_queue.size()) {
        const size_t curr_idx = dead_end_queue[head++];
        const int r = static_cast<int>(curr_idx / this->width);
        const int c = static_cast<int>(curr_idx % this->width);

        this->visited[r][c] = true;

        for (int i = 0; i < 4; ++i) {
            const int nr = r + d_row[i];
            const int nc = c + d_col[i];

            if (nr < 0 || nr >= this->height || nc < 0 || nc >= this->width || grid[nr][nc] == WALL) {
                continue;
            }

            const size_t neighbor_idx = static_cast<size_t>(nr) * this->width + nc;
            if (degree[neighbor_idx] == PRUNED) {
                continue;
            }

            if (degree[neighbor_idx] > 0) {
                --degree[neighbor_idx];
            }

            const bool is_start_or_end = (nr == this->start.first && nc == this->start.second) ||
                                         (nr == this->end.first && nc == this->end.second);

            if (!is_start_or_end && degree[neighbor_idx] <= 1) {
                degree[neighbor_idx] = PRUNED;
                dead_end_queue.push_back(neighbor_idx);
            }
        }
    }

    // Pass 3: Surviving PATH cells form the exact solution path
    size_t solution_cell_count = 0;
    for (int r = 0; r < this->height; ++r) {
        for (int c = 0; c < this->width; ++c) {
            const size_t idx = static_cast<size_t>(r) * this->width + c;
            if (grid[r][c] == PATH && degree[idx] != PRUNED) {
                this->solution[r][c] = true;
                ++solution_cell_count;
            }
        }
    }

    return (solution_cell_count > 0 &&
            this->solution[this->start.first][this->start.second] &&
            this->solution[this->end.first][this->end.second]);
}
