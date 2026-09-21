#include "BidirectionalGreedyBestFirstSolver.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <cstdint>

struct BidirNode {
    int r, c;
    int h;

    bool operator>(const BidirNode& other) const {
        return h > other.h;
    }
};

inline static int manhattan(const int r1, const int c1, const int r2, const int c2) {
    return std::abs(r1 - r2) + std::abs(c1 - c2);
}

bool BidirectionalGreedyBestFirstSolver::solve(const Maze &maze_object) {
    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();

    this->visited.assign(this->height, std::vector<bool>(this->width, false));
    this->solution.assign(this->height, std::vector<bool>(this->width, false));

    const size_t total_cells = static_cast<size_t>(this->height) * this->width;
    this->visited_count = 0;

    // Flat 1D state array:
    // 0xFF: unvisited
    // 0x00 - 0x03: visited by Forward (bits 0-1 = direction: UP=0, RIGHT=1, DOWN=2, LEFT=3)
    // 0x80 - 0x83: visited by Backward (bit 7 set, bits 0-1 = direction)
    constexpr uint8_t UNVISITED = 0xFF;
    constexpr uint8_t BACKWARD_FLAG = 0x80;
    constexpr uint8_t DIR_MASK = 0x03;

    std::vector<uint8_t> cell_state(total_cells, UNVISITED);

    // Movement deltas matching Direction enum: UP=0, RIGHT=1, DOWN=2, LEFT=3
    constexpr int d_row[4] = {-1, 0, 1, 0};
    constexpr int d_col[4] = {0, 1, 0, -1};

    // Forward and Backward min-priority queues
    std::priority_queue<BidirNode, std::vector<BidirNode>, std::greater<BidirNode>> open_forward;
    std::priority_queue<BidirNode, std::vector<BidirNode>, std::greater<BidirNode>> open_backward;

    // Initialize Forward frontier at start
    const size_t start_idx = static_cast<size_t>(this->start.first) * this->width + this->start.second;
    cell_state[start_idx] = 0x00; // Forward root
    this->visited[this->start.first][this->start.second] = true;
    open_forward.push({this->start.first, this->start.second,
                       manhattan(this->start.first, this->start.second, this->end.first, this->end.second)});
    ++this->visited_count;

    // Initialize Backward frontier at end
    const size_t end_idx = static_cast<size_t>(this->end.first) * this->width + this->end.second;
    cell_state[end_idx] = BACKWARD_FLAG; // Backward root
    this->visited[this->end.first][this->end.second] = true;
    open_backward.push({this->end.first, this->end.second,
                        manhattan(this->end.first, this->end.second, this->start.first, this->start.second)});
    ++this->visited_count;

    // Meeting points
    int meet_forward_r = -1, meet_forward_c = -1;
    int meet_backward_r = -1, meet_backward_c = -1;
    bool collision_found = false;

    // Main bidirectional expansion loop
    while (!open_forward.empty() && !open_backward.empty()) {
        // --- Step 1: Expand one node from Forward frontier ---
        {
            const auto [r, c, h] = open_forward.top();
            open_forward.pop();

            for (int i = 0; i < 4; ++i) {
                const int nr = r + d_row[i];
                const int nc = c + d_col[i];

                if (nr >= 0 && nr < this->height && nc >= 0 && nc < this->width && grid[nr][nc] == PATH) {
                    const size_t nidx = static_cast<size_t>(nr) * this->width + nc;
                    const uint8_t state = cell_state[nidx];

                    // Check if Backward frontier already visited this neighbor
                    if (state != UNVISITED && (state & BACKWARD_FLAG) != 0) {
                        meet_forward_r = r;
                        meet_forward_c = c;
                        meet_backward_r = nr;
                        meet_backward_c = nc;
                        collision_found = true;
                        break;
                    }

                    if (state == UNVISITED) {
                        cell_state[nidx] = static_cast<uint8_t>(i); // Forward direction
                        this->visited[nr][nc] = true;
                        ++this->visited_count;
                        const int nh = manhattan(nr, nc, this->end.first, this->end.second);
                        open_forward.push({nr, nc, nh});
                    }
                }
            }

            if (collision_found) {
                break;
            }
        }

        // --- Step 2: Expand one node from Backward frontier ---
        {
            const auto [r, c, h] = open_backward.top();
            open_backward.pop();

            for (int i = 0; i < 4; ++i) {
                const int nr = r + d_row[i];
                const int nc = c + d_col[i];

                if (nr >= 0 && nr < this->height && nc >= 0 && nc < this->width && grid[nr][nc] == PATH) {
                    const size_t nidx = static_cast<size_t>(nr) * this->width + nc;
                    const uint8_t state = cell_state[nidx];

                    // Check if Forward frontier already visited this neighbor
                    if (state != UNVISITED && (state & BACKWARD_FLAG) == 0) {
                        meet_forward_r = nr;
                        meet_forward_c = nc;
                        meet_backward_r = r;
                        meet_backward_c = c;
                        collision_found = true;
                        break;
                    }

                    if (state == UNVISITED) {
                        cell_state[nidx] = static_cast<uint8_t>(BACKWARD_FLAG | i); // Backward direction
                        this->visited[nr][nc] = true;
                        ++this->visited_count;
                        const int nh = manhattan(nr, nc, this->start.first, this->start.second);
                        open_backward.push({nr, nc, nh});
                    }
                }
            }

            if (collision_found) {
                break;
            }
        }
    }

    if (!collision_found) {
        return false;
    }

    // --- Step 3: Reconstruct Solution Path in O(L) ---
    // Trace from meet_forward back to start
    int curr_r = meet_forward_r;
    int curr_c = meet_forward_c;
    while (curr_r != this->start.first || curr_c != this->start.second) {
        this->solution[curr_r][curr_c] = true;
        const size_t idx = static_cast<size_t>(curr_r) * this->width + curr_c;
        const uint8_t dir = cell_state[idx] & DIR_MASK;
        curr_r -= d_row[dir];
        curr_c -= d_col[dir];
    }
    this->solution[this->start.first][this->start.second] = true;

    // Trace from meet_backward back to end
    curr_r = meet_backward_r;
    curr_c = meet_backward_c;
    while (curr_r != this->end.first || curr_c != this->end.second) {
        this->solution[curr_r][curr_c] = true;
        const size_t idx = static_cast<size_t>(curr_r) * this->width + curr_c;
        const uint8_t dir = cell_state[idx] & DIR_MASK;
        curr_r -= d_row[dir];
        curr_c -= d_col[dir];
    }
    this->solution[this->end.first][this->end.second] = true;

    return true;
}
