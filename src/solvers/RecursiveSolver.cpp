/**
 * =============================================================================
 * ITERATIVE DEPTH-FIRST SEARCH (DFS) SOLVER ("RECURSIVE")
 * =============================================================================
 *
 * 1. ALGORITHM STRATEGY:
 *    - Depth-First Traversal with Backtracking: Explores each corridor deeply until
 *      encountering a junction, dead end, or the destination. Upon hitting a dead end,
 *      backtracks along the path stack to the last branch with unvisited neighbors.
 *
 * 2. STACK OVERFLOW PREVENTION (ITERATIVE VS RECURSIVE):
 *    - Although named "RecursiveSolver" for historical reasons, the algorithm is
 *      implemented iteratively using an explicit heap-allocated stack.
 *    - On large mazes (e.g. 131,073 x 131,073), solution paths can exceed millions
 *      of steps. Function call-stack recursion would instantly exhaust the standard
 *      OS thread stack limit (typically 8 MB) and trigger a fatal SIGSEGV.
 *
 * 3. ADAPTIVE CONTAINER PRE-ALLOCATION:
 *    - By default, std::stack uses std::deque, which allocates nodes in fragmented
 *      chunks (typically 512 bytes).
 *    - We use std::stack<Direction, std::vector<Direction>> with an adaptive reserve:
 *      cap = min(total_cells / 4, max(1024, (H + W) * 2))
 *      derived from the theoretical spanning tree diameter.
 *    - This provides contiguous memory layout, eliminates page allocation stalls,
 *      and ensures optimal CPU memory locality.
 * =============================================================================
 */

#include "RecursiveSolver.hpp"
#include <vector>
#include <stack>
#include <algorithm>

bool RecursiveSolver::solve(const Maze &maze_object) { //dfs

    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();

    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &maze = maze_object.getMaze();

    this->visited.assign(this->height, std::vector<bool>(this->width, false));
    this->solution.assign(this->height, std::vector<bool>(this->width, false));

    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

    // Adaptive stack container sizing:
    // Sized based on theoretical spanning tree diameter. Pre-allocating capacity
    // eliminates repeated dynamic reallocations during deep traversal.
    const size_t adaptive_stack_cap = std::min<size_t>(
        total_cells / 4,
        std::max<size_t>(1024ULL, static_cast<size_t>(this->height + this->width) * 2)
    );

    std::vector<Direction> step_container;
    step_container.reserve(adaptive_stack_cap);
    std::stack<Direction, std::vector<Direction>> steps(std::move(step_container));

    constexpr Direction directions[] = {UP, RIGHT, DOWN, LEFT};

    std::pair <int, int> current = start;
    visited[current.first][current.second] = true;

    int lastStep = -1;


    //Search end
    while (current != end && !(lastStep == N_DIRECTIONS-1 && current == start)) {

        bool stepped = false;
        int d = lastStep+1;
        while (d <= N_DIRECTIONS && !stepped) {
            if (d == N_DIRECTIONS) { //backtrack
                lastStep = steps.top();
                steps.pop();
                switch (lastStep) {
                    case UP: current.first = current.first+1; break;
                    case RIGHT: current.second = current.second-1; break;
                    case DOWN: current.first = current.first-1; break;
                    case LEFT: current.second = current.second+1; break;
                    default: break;
                }
            } else {
                switch (directions[d]) {
                    case UP:
                        if (current.first > 0 && maze[current.first-1][current.second] == PATH && !visited[current.first-1][current.second]) {
                            steps.push(UP);
                            current.first = current.first-1;
                            visited[current.first][current.second] = true;
                            lastStep = -1;
                            stepped = true;
                            //std::cout << "go UP" << std::endl;
                        }
                        break;
                    case RIGHT:
                        if (current.second < width-1 && maze[current.first][current.second+1] == PATH && !visited[current.first][current.second+1]) {
                            steps.push(RIGHT);
                            current.second = current.second+1;
                            visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go RIGHT" << std::endl;
                        }
                        break;
                    case DOWN:
                        if (current.first < height-1 && maze[current.first+1][current.second] == PATH && !visited[current.first+1][current.second]) {
                            steps.push(DOWN);
                            current.first = current.first+1;
                            visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go DOWN" << std::endl;
                        }
                        break;
                    case LEFT:
                        if (current.second > 0 && maze[current.first][current.second-1] == PATH && !visited[current.first][current.second-1]) {
                            steps.push(LEFT);
                            current.second = current.second-1;
                            visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go LEFT" << std::endl;
                        }
                        break;
                }
            }
            ++d;
        }

    }

    //Build solution
    if (current == end) {
        solution[current.first][current.second] = true;
        while (!steps.empty()) {
            lastStep = steps.top();
            steps.pop();
            switch (lastStep) {
                case UP: current.first = current.first+1; break;
                case RIGHT: current.second = current.second-1; break;
                case DOWN: current.first = current.first-1; break;
                case LEFT: current.second = current.second+1; break;
                default: break;
            }
            solution[current.first][current.second] = true;
        }
        return true;
    }

    return false;

}
