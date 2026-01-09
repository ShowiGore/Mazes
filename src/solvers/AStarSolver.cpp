#include "AStarSolver.hpp"
#include <queue>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

struct AStarNode {
    int r, c;     // Coordinates
    int f_score;  // f = g + h

    bool operator>(const AStarNode& other) const {
        return f_score > other.f_score;
    }
};

inline static int heuristic(const int r, const int c, const int endR, const int endC) {
    return std::abs(r - endR) + std::abs(c - endC);
}

template <bool IsWeighted>
bool AStarSolver::solve_impl(const Maze &maze_object) { // A*
    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &maze = maze_object.getMaze();

    // Reset solution and visited grids
    this->solution.assign(this->height, std::vector<bool>(this->width, false));
    this->visited.assign(this->height, std::vector<bool>(this->width, false));

    // Data structures for A*
    // g_score stores the cost of the cheapest path from start to node currently known
    std::vector<std::vector<int>> g_score(this->height, std::vector<int>(this->width, std::numeric_limits<int>::max()));

    // Parent grid to reconstruct the path
    std::vector<std::vector<std::pair<int, int>>> parent(this->height, std::vector<std::pair<int, int>>(this->width, {-1, -1}));

    // Open Set: Min-priority queue based on f_score
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set;

    // Initialize start node
    const int start_h = heuristic(start.first, start.second, end.first, end.second);
    int start_f;
    if constexpr (IsWeighted) { // Apply weight to initial node too for consistency
        start_f = start_h * heuristic_weight;
    } else {
        start_f = start_h;
    }
    open_set.push({start.first, start.second, start_f});
    g_score[start.first][start.second] = 0;

    // Standard directions deltas: UP, RIGHT, DOWN, LEFT

    while (!open_set.empty()) {

        // Get the node with the lowest f_score
        auto [r, c, f_score] = open_set.top();
        open_set.pop();

        if (r == end.first && c == end.second) { // If we reached the end, reconstruction begins
            std::pair<int, int> path_node = end;

            while (path_node.first != -1) { // Backtrack from end to start using the parent grid
                solution[path_node.first][path_node.second] = true;
                path_node = parent[path_node.first][path_node.second];
            }
            return true;
        }


        const int current_h = heuristic(r, c, end.first, end.second);
        int current_g;
        if constexpr (IsWeighted) {
            current_g = f_score - (current_h * heuristic_weight);
        } else {
            current_g = f_score - current_h;
        }

        if (current_g > g_score[r][c]) { // Skip if we already found a better path to this node
            continue;
        }

        visited[r][c] = true; // Mark as visited (Closed Set equivalent)

        // Explore neighbors
        for (int i = 0; i < 4; ++i) {
            constexpr int dc[] = {0, 1, 0, -1};
            constexpr int dr[] = {-1, 0, 1, 0};
            const int nr = r + dr[i];
            const int nc = c + dc[i];

            if (nr >= 0 && nr < height && nc >= 0 && nc < width && maze[nr][nc] == PATH) { // Check boundaries and walls

                const int tentative_g = g_score[r][c] + 1; // Distance between neighbors is always 1

                if (tentative_g < g_score[nr][nc]) { // If this path to neighbor is better than any previous one
                    parent[nr][nc] = {r, c};
                    g_score[nr][nc] = tentative_g;

                    const int h = heuristic(nr, nc, end.first, end.second);

                    int f;
                    if constexpr (IsWeighted) {
                        f = tentative_g + (h * heuristic_weight);
                    } else {
                        f = tentative_g + h;
                    }

                    open_set.push({nr, nc, f});
                }
            }
        }
    }

    return false; // No solution found
}

bool AStarSolver::solve(const Maze &maze_object) {
    if (this->heuristic_weight == 1) { // Dispatcher to select implementation
        return solve_impl<false>(maze_object);
    } else {
        return solve_impl<true>(maze_object);
    }
}
