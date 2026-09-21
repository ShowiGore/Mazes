/**
 * =============================================================================
 * GREEDY BEST-FIRST SEARCH (GBFS) SOLVER
 * =============================================================================
 *
 * 1. ALGORITHM STRATEGY:
 *    - Heuristic Priority Search: Explores candidate path nodes in order of
 *      minimum estimated distance to the destination, using the L1 Manhattan metric:
 *          h(r, c) = |r - end_r| + |c - end_c|
 *    - In a 4-connected planar grid, the L1 metric represents the geodesic distance
 *      lower-bound in the absence of obstacles, providing optimal search guidance.
 *    - By focusing entirely on h without accumulating step costs (unlike A*), GBFS
 *      dramatically accelerates convergence towards the goal in large tree mazes.
 *
 * 2. ADAPTIVE CONTAINER ALLOCATION:
 *    - The priority queue backing vector is pre-allocated with an adaptive capacity:
 *      cap = min(total_cells / 2, max(1024, (H + W) * 4))
 *    - This eliminates dynamic vector growth overhead during frontier exploration.
 *
 * 3. 1-BYTE DIRECTION-COMPRESSED PARENT TRACKING:
 *    - Replaces standard 8-byte coordinate pairs with a 1-byte direction (0..3)
 *      in a flat 1D array. Backtracking from destination to start runs in exact O(L) time.
 * =============================================================================
 */

#include "GreedyBestFirstSolver.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

struct GBFSNode {
    int r, c;
    int h;

    bool operator>(const GBFSNode& other) const {
        return h > other.h;
    }
};

inline static int manhattan(const int r1, const int c1, const int r2, const int c2) {
    return std::abs(r1 - r2) + std::abs(c1 - c2);
}

bool GreedyBestFirstSolver::solve(const Maze &maze_object) {
    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();

    this->visited.assign(this->height, std::vector<bool>(this->width, false));
    this->solution.assign(this->height, std::vector<bool>(this->width, false));

    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

    // Flat 1D parent direction array (1 byte per cell):
    // 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT, 0xFF = unvisited/none.
    std::vector<uint8_t> parent_dir(total_cells, 0xFF);

    // Movement deltas matching Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3
    constexpr int d_row[4] = {-1, 0, 1, 0};
    constexpr int d_col[4] = {0, 1, 0, -1};

    // Adaptive frontier container sizing:
    // Pre-allocates priority queue capacity to eliminate reallocation latency.
    const size_t adaptive_frontier_cap = std::min<size_t>(
        total_cells / 2,
        std::max<size_t>(1024ULL, static_cast<size_t>(this->height + this->width) * 4)
    );

    std::vector<GBFSNode> open_container;
    open_container.reserve(adaptive_frontier_cap);
    std::priority_queue<GBFSNode, std::vector<GBFSNode>, std::greater<GBFSNode>> open_set(
        std::greater<GBFSNode>(), std::move(open_container)
    );

    const int start_h = manhattan(this->start.first, this->start.second, this->end.first, this->end.second);
    open_set.push({this->start.first, this->start.second, start_h});
    this->visited[this->start.first][this->start.second] = true;

    bool reached_end = false;

    while (!open_set.empty()) {
        const auto [r, c, h] = open_set.top();
        open_set.pop();

        if (r == this->end.first && c == this->end.second) {
            reached_end = true;
            break;
        }

        for (int i = 0; i < 4; ++i) {
            const int nr = r + d_row[i];
            const int nc = c + d_col[i];

            if (nr >= 0 && nr < this->height && nc >= 0 && nc < this->width && grid[nr][nc] == PATH) {
                const size_t nidx = static_cast<size_t>(nr) * this->width + nc;
                if (!this->visited[nr][nc]) {
                    this->visited[nr][nc] = true;
                    parent_dir[nidx] = static_cast<uint8_t>(i);

                    const int nh = manhattan(nr, nc, this->end.first, this->end.second);
                    open_set.push({nr, nc, nh});
                }
            }
        }
    }

    if (!reached_end) {
        return false;
    }

    // Reconstruct solution path by backtracking using 1-byte parent_dir
    int curr_r = this->end.first;
    int curr_c = this->end.second;

    while (curr_r != this->start.first || curr_c != this->start.second) {
        this->solution[curr_r][curr_c] = true;
        const size_t idx = static_cast<size_t>(curr_r) * this->width + curr_c;
        const uint8_t dir = parent_dir[idx];

        curr_r -= d_row[dir];
        curr_c -= d_col[dir];
    }
    this->solution[this->start.first][this->start.second] = true;

    return true;
}
