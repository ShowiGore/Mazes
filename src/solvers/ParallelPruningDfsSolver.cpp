#include "ParallelPruningDfsSolver.hpp"

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

ParallelPruningDfsSolver::ParallelPruningDfsSolver() {
    this->solver_name = "cpu-prune-dfs";
}

bool ParallelPruningDfsSolver::solve(const Maze &maze) {
    this->height = maze.getHeight();
    this->width = maze.getWidth();
    this->start = maze.getStart();
    this->end = maze.getEnd();

    const int H = this->height;
    const int W = this->width;
    const int words_per_row = (W + 63) / 64;
    const size_t total_words = static_cast<size_t>(H) * words_per_row;

    // =========================================================================
    // STEP 1: Bitpack Maze Grid into 64-bit SIMD Words
    // 1ULL = OPEN PATH, 0ULL = WALL
    // =========================================================================
    const auto &maze_grid = maze.getMaze();
    std::vector<uint64_t> grid_A(total_words, 0ULL);
    std::vector<uint64_t> grid_B(total_words, 0ULL);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < H; ++r) {
        const size_t row_offset = static_cast<size_t>(r) * words_per_row;
        for (int c = 0; c < W; ++c) {
            if (!maze_grid[r][c]) { // false in Maze means open path
                grid_A[row_offset + (c / 64)] |= (1ULL << (c % 64));
            }
        }
    }

    // =========================================================================
    // STEP 2: Phase 1 - Multi-Threaded 64-bit SIMD Dead-End Pruning
    // =========================================================================
    uint64_t *grid_in = grid_A.data();
    uint64_t *grid_out = grid_B.data();

    this->prune_passes = 0;
    this->total_pruned = 0;

    const uint64_t last_word_mask = (W % 64 != 0) ? ((1ULL << (W % 64)) - 1ULL) : ~0ULL;

    for (int pass = 0; pass < max_prune_iterations; ++pass) {
        size_t pass_pruned = 0;

        // Preserve perimeter rows (row 0 and row H-1 containing start and end)
        for (int w = 0; w < words_per_row; ++w) {
            grid_out[w] = grid_in[w];
            grid_out[static_cast<size_t>(H - 1) * words_per_row + w] = grid_in[static_cast<size_t>(H - 1) * words_per_row + w];
        }

        #pragma omp parallel for reduction(+:pass_pruned) schedule(static)
        for (int r = 1; r < H - 1; ++r) {
            const size_t row_offset = static_cast<size_t>(r) * words_per_row;
            const size_t north_offset = static_cast<size_t>(r - 1) * words_per_row;
            const size_t south_offset = static_cast<size_t>(r + 1) * words_per_row;

            for (int w = 0; w < words_per_row; ++w) {
                const uint64_t curr = grid_in[row_offset + w];
                if (curr == 0ULL) {
                    grid_out[row_offset + w] = 0ULL;
                    continue;
                }

                const uint64_t north = grid_in[north_offset + w];
                const uint64_t south = grid_in[south_offset + w];

                const uint64_t prev_word = (w > 0) ? grid_in[row_offset + (w - 1)] : 0ULL;
                const uint64_t next_word = (w + 1 < words_per_row) ? grid_in[row_offset + (w + 1)] : 0ULL;

                // West neighbor: column c-1 (shift left)
                // East neighbor: column c+1 (shift right)
                const uint64_t west = (curr << 1) | (prev_word >> 63);
                const uint64_t east = (curr >> 1) | (next_word << 63);

                // 4-bitplane parallel full adder
                const uint64_t sum1 = north ^ south;
                const uint64_t carry1 = north & south;

                const uint64_t sum2 = west ^ east;
                const uint64_t carry2 = west & east;

                const uint64_t carry3 = sum1 & sum2;

                // sum >= 2 iff any carry is set
                const uint64_t sum_ge_2 = carry1 | carry2 | carry3;

                // Candidates to prune: open cell (curr == 1) with <= 1 open neighbor
                uint64_t prune_mask = curr & (~sum_ge_2);

                // Protect Start and End
                if (r == this->start.first && w == (this->start.second / 64)) {
                    prune_mask &= ~(1ULL << (this->start.second % 64));
                }
                if (r == this->end.first && w == (this->end.second / 64)) {
                    prune_mask &= ~(1ULL << (this->end.second % 64));
                }

                // Mask out-of-bounds bits on the last word
                if (w == words_per_row - 1) {
                    prune_mask &= last_word_mask;
                }

                if (prune_mask != 0) {
                    pass_pruned += __builtin_popcountll(prune_mask);
                }

                grid_out[row_offset + w] = curr & (~prune_mask);
            }
        }

        this->total_pruned += pass_pruned;
        this->prune_passes++;

        std::swap(grid_in, grid_out);

        if (pass_pruned == 0) {
            break; // Pruning fully converged!
        }
    }

    // `grid_in` points to the latest pruned maze
    const uint64_t *pruned_grid = grid_in;

    // =========================================================================
    // STEP 3: Phase 2 - In-Place Bidirectional DFS with Manhattan Ordering
    // =========================================================================
    std::vector<uint64_t> visited_fwd(total_words, 0ULL);
    std::vector<uint64_t> visited_bwd(total_words, 0ULL);

    std::vector<std::pair<int, int>> stack_fwd;
    std::vector<std::pair<int, int>> stack_bwd;
    stack_fwd.reserve(65536);
    stack_bwd.reserve(65536);

    const auto set_visited = [&](std::vector<uint64_t> &visited_vec, int r, int c) {
        const size_t idx = static_cast<size_t>(r) * words_per_row + (c / 64);
        visited_vec[idx] |= (1ULL << (c % 64));
    };

    const auto is_visited = [&](const std::vector<uint64_t> &visited_vec, int r, int c) -> bool {
        const size_t idx = static_cast<size_t>(r) * words_per_row + (c / 64);
        return (visited_vec[idx] >> (c % 64)) & 1ULL;
    };

    stack_fwd.push_back(this->start);
    set_visited(visited_fwd, this->start.first, this->start.second);

    stack_bwd.push_back(this->end);
    set_visited(visited_bwd, this->end.first, this->end.second);

    size_t fwd_visited_count = 1;
    size_t bwd_visited_count = 1;

    // Fast inline neighbor search
    struct Neighbor {
        int r, c, dist;
    };

    auto get_best_neighbors = [&](int r, int c, int target_r, int target_c,
                                  const std::vector<uint64_t> &visited_self) {
        std::array<Neighbor, 4> neighbors;
        int count = 0;

        constexpr int dr[4] = {-1, 1, 0, 0};
        constexpr int dc[4] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i) {
            const int nr = r + dr[i];
            const int nc = c + dc[i];

            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;

            const size_t idx = static_cast<size_t>(nr) * words_per_row + (nc / 64);
            const uint64_t bit = 1ULL << (nc % 64);

            if ((pruned_grid[idx] & bit) && !(visited_self[idx] & bit)) {
                const int dist = std::abs(nr - target_r) + std::abs(nc - target_c);
                neighbors[count++] = {nr, nc, dist};
            }
        }

        // Sort ascending by Manhattan distance to target
        for (int i = 1; i < count; ++i) {
            Neighbor key = neighbors[i];
            int j = i - 1;
            while (j >= 0 && neighbors[j].dist > key.dist) {
                neighbors[j + 1] = neighbors[j];
                --j;
            }
            neighbors[j + 1] = key;
        }

        return std::make_pair(neighbors, count);
    };

    bool collision = false;
    std::pair<int, int> meet_cell;

    while (!stack_fwd.empty() && !stack_bwd.empty()) {
        // --- Forward Step ---
        {
            const auto [r, c] = stack_fwd.back();

            if (is_visited(visited_bwd, r, c)) {
                meet_cell = {r, c};
                collision = true;
                break;
            }

            auto [nbrs, count] = get_best_neighbors(r, c, this->end.first, this->end.second, visited_fwd);
            if (count == 0) {
                stack_fwd.pop_back(); // Backtrack dead end
            } else {
                const auto &best = nbrs[0];
                set_visited(visited_fwd, best.r, best.c);
                stack_fwd.push_back({best.r, best.c});
                fwd_visited_count++;

                if (is_visited(visited_bwd, best.r, best.c)) {
                    meet_cell = {best.r, best.c};
                    collision = true;
                    break;
                }
            }
        }

        // --- Backward Step ---
        {
            const auto [r, c] = stack_bwd.back();

            if (is_visited(visited_fwd, r, c)) {
                meet_cell = {r, c};
                collision = true;
                break;
            }

            auto [nbrs, count] = get_best_neighbors(r, c, this->start.first, this->start.second, visited_bwd);
            if (count == 0) {
                stack_bwd.pop_back(); // Backtrack dead end
            } else {
                const auto &best = nbrs[0];
                set_visited(visited_bwd, best.r, best.c);
                stack_bwd.push_back({best.r, best.c});
                bwd_visited_count++;

                if (is_visited(visited_fwd, best.r, best.c)) {
                    meet_cell = {best.r, best.c};
                    collision = true;
                    break;
                }
            }
        }
    }

    if (!collision) {
        std::cerr << "[ParallelPruningDfsSolver] Search exhausted without collision!\n";
        return false;
    }

    this->visited_count = fwd_visited_count + bwd_visited_count;

    // =========================================================================
    // STEP 4: Path Reconstruction
    // =========================================================================
    size_t fwd_idx = stack_fwd.size() - 1;
    while (fwd_idx > 0 && stack_fwd[fwd_idx] != meet_cell) {
        --fwd_idx;
    }

    size_t bwd_idx = stack_bwd.size() - 1;
    while (bwd_idx > 0 && stack_bwd[bwd_idx] != meet_cell) {
        --bwd_idx;
    }

    this->solution.assign(H, std::vector<bool>(W, false));
    this->visited.assign(H, std::vector<bool>(W, false));

    size_t solution_cells = 0;

    // Forward path from start up to meet_cell
    for (size_t i = 0; i <= fwd_idx; ++i) {
        const auto [r, c] = stack_fwd[i];
        this->solution[r][c] = true;
        solution_cells++;
    }

    // Backward path from meet_cell down to end
    if (bwd_idx > 0) {
        for (size_t i = bwd_idx - 1; ; --i) {
            const auto [r, c] = stack_bwd[i];
            this->solution[r][c] = true;
            solution_cells++;
            if (i == 0) break;
        }
    }

    // Mark visited cells for visualization
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < H; ++r) {
        const size_t row_offset = static_cast<size_t>(r) * words_per_row;
        for (int c = 0; c < W; ++c) {
            const size_t idx = row_offset + (c / 64);
            const uint64_t bit = 1ULL << (c % 64);
            if ((visited_fwd[idx] & bit) || (visited_bwd[idx] & bit)) {
                this->visited[r][c] = true;
            }
        }
    }

    std::cout << "[ParallelPruningDfsSolver] Pruned " << this->total_pruned
              << " dead ends in " << this->prune_passes << " passes. "
              << "Explored " << this->visited_count << " cells. "
              << "Solution length: " << solution_cells << " cells.\n";

    return (solution_cells > 0 &&
            this->solution[this->start.first][this->start.second] &&
            this->solution[this->end.first][this->end.second]);
}
