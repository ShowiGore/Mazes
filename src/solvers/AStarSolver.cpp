/**
 * =============================================================================
 * A* SEARCH SOLVER (STANDARD AND WEIGHTED)
 * =============================================================================
 *
 * 1. ALGORITHM STRATEGY:
 *    - Heuristic Search with Cost Balancing: Evaluates nodes using f(n) = g(n) + w * h(n),
 *      where g(n) is the exact cumulative step cost from start, and h(n) is the
 *      L1 Manhattan distance heuristic:
 *          h(r, c) = |r - end_r| + |c - end_c|
 *
 * 2. THEORETICAL OPTIMALITY GUARANTEES:
 *    - Admissibility: On a 4-connected grid with uniform step cost c(u, v) = 1,
 *      the Manhattan distance is strictly admissible (h(n) <= d*(n, goal)) because
 *      no path can reach the goal in fewer steps than the orthogonal coordinate delta.
 *    - Consistency (Monotonicity): The heuristic satisfies the triangle inequality:
 *          h(u) <= c(u, v) + h(v) = 1 + h(v)
 *      Consistency guarantees that the first time any node is extracted from the open
 *      set, its g_score is guaranteed to be optimal. No closed node re-expansion is needed.
 *    - Weighted A* Trade-off: When heuristic_weight w > 1, the algorithm becomes
 *      epsilon-admissible (solution cost <= w * C*), dramatically reducing the number of
 *      expanded states by biasing search progress aggressively toward the goal.
 *
 * 3. ADAPTIVE CONTAINER PRE-ALLOCATION:
 *    - The open_set priority queue backing vector is pre-allocated adaptively:
 *      cap = min(total_cells / 2, max(1024, (H + W) * 4))
 *    - This eliminates memory reallocation stalls during frontier expansion.
 * =============================================================================
 */

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

    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

    // Data structures for A*
    // g_score stores the cost of the cheapest path from start to node currently known
    std::vector<std::vector<int>> g_score(this->height, std::vector<int>(this->width, std::numeric_limits<int>::max()));

    // Parent grid to reconstruct the path
    std::vector<std::vector<std::pair<int, int>>> parent(this->height, std::vector<std::pair<int, int>>(this->width, {-1, -1}));

    // Adaptive open set container sizing:
    // Sized to accommodate the frontier expansion without frequent reallocations.
    const size_t adaptive_frontier_cap = std::min<size_t>(
        total_cells / 2,
        std::max<size_t>(1024ULL, static_cast<size_t>(this->height + this->width) * 4)
    );

    std::vector<AStarNode> open_container;
    open_container.reserve(adaptive_frontier_cap);
    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open_set(
        std::greater<AStarNode>(), std::move(open_container)
    );

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
