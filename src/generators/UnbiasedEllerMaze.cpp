#include "UnbiasedEllerMaze.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>

// =============================================================================
// HIGH-PERFORMANCE DISJOINT-SET (UNION-FIND) WITH PATH COMPRESSION & RANK
// =============================================================================
struct FastDisjointSet {
    std::vector<int> parent;
    std::vector<uint8_t> rank;

    explicit FastDisjointSet(size_t n) : parent(n), rank(n, 0) {
        reset(n);
    }

    void reset(size_t n) {
        std::iota(parent.begin(), parent.end(), 0);
        std::fill(rank.begin(), rank.end(), 0);
    }

    int find(int i) {
        int root = i;
        while (parent[root] != root) {
            root = parent[root];
        }
        // Two-pass path compression
        int curr = i;
        while (curr != root) {
            int nxt = parent[curr];
            parent[curr] = root;
            curr = nxt;
        }
        return root;
    }

    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i == root_j) return false;

        if (rank[root_i] < rank[root_j]) {
            parent[root_i] = root_j;
        } else if (rank[root_i] > rank[root_j]) {
            parent[root_j] = root_i;
        } else {
            parent[root_j] = root_i;
            rank[root_i]++;
        }
        return true;
    }
};

UnbiasedEllerMaze::UnbiasedEllerMaze(const int height, const int width, const unsigned int seed)
    : Maze(height, width, seed) {
    this->generator_name = "eller";
    UnbiasedEllerMaze::init();
    UnbiasedEllerMaze::generate();
    buildStartEnd();
}

UnbiasedEllerMaze::UnbiasedEllerMaze(const int height, const int width)
    : Maze(height, width) {
    this->generator_name = "eller";
    UnbiasedEllerMaze::init();
    UnbiasedEllerMaze::generate();
    buildStartEnd();
}

void UnbiasedEllerMaze::init() {
    for (int h = 0; h < this->height; ++h) {
        for (int w = 0; w < this->width; ++w) {
            maze[h][w] = WALL;
        }
    }
}

void UnbiasedEllerMaze::generate() {
    const size_t num_cell_rows = static_cast<size_t>(height) / 2;
    const size_t num_cell_cols = static_cast<size_t>(width) / 2;

    if (num_cell_rows == 0 || num_cell_cols == 0) {
        return;
    }

    // Disjoint set for the current row
    FastDisjointSet uf(num_cell_cols);

    // Flat pre-allocated intrusive linked-list buffers for O(C) grouping with zero heap allocation
    std::vector<int> head(num_cell_cols, -1);
    std::vector<int> next_col(num_cell_cols, -1);
    std::vector<int> set_size(num_cell_cols, 0);
    std::vector<int> active_roots;
    active_roots.reserve(num_cell_cols);
    std::vector<int> col_buffer;
    col_buffer.reserve(num_cell_cols);

    // State array for columns that received a vertical drop from the row above
    // Stores the incoming token index; -1 means unvisited / singleton.
    std::vector<int> incoming_drop_token(num_cell_cols, -1);
    std::vector<int> drop_leader(num_cell_cols, -1);

    // Candidate horizontal indices for randomized 1D Kruskal pass
    std::vector<size_t> h_candidates(num_cell_cols > 1 ? num_cell_cols - 1 : 0);
    std::iota(h_candidates.begin(), h_candidates.end(), 0);

    // Number of transition rows for progressive hierarchical merge near bottom
    const size_t K_MERGE_ROWS = std::min<size_t>(16, std::max<size_t>(4, num_cell_rows / 8));

    // Fast PRNG helper for double in [0, 1)
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);

    // Closed-loop PI controller parameters for exact 50/50 horizontal/vertical edge balance
    const double target_h_per_row = static_cast<double>(num_cell_cols - 1) * 0.50;
    double cum_target_h = 0.0;
    double cum_actual_h = 0.0;

    for (size_t r = 0; r < num_cell_rows; ++r) {
        const bool is_last_row = (r == num_cell_rows - 1);
        const bool is_near_bottom = (r + K_MERGE_ROWS >= num_cell_rows);

        cum_target_h += target_h_per_row;

        // Reset Union-Find for this row
        uf.reset(num_cell_cols);

        // 1. Carve room paths and connect columns that dropped from the SAME set in the previous row
        if (r > 0) {
            std::fill(drop_leader.begin(), drop_leader.end(), -1);
            for (size_t c = 0; c < num_cell_cols; ++c) {
                maze[2 * r + 1][2 * c + 1] = PATH; // Carve room
                const int token = incoming_drop_token[c];
                if (token >= 0) {
                    if (drop_leader[token] == -1) {
                        drop_leader[token] = static_cast<int>(c);
                    } else {
                        uf.unite(drop_leader[token], static_cast<int>(c));
                    }
                }
            }
        } else {
            for (size_t c = 0; c < num_cell_cols; ++c) {
                maze[2 * r + 1][2 * c + 1] = PATH; // Carve room
            }
        }

        // 2. Horizontal Carving Phase (Randomized 1D Kruskal pass with Closed-Loop PI Control)
        const double error = (cum_target_h - cum_actual_h) / static_cast<double>(num_cell_cols);
        double p_horizontal = std::clamp(0.55 + 0.35 * error, 0.30, 0.85);

        if (is_last_row) {
            p_horizontal = 1.0; // Force merge on last row to connect all disjoint sets
        } else if (is_near_bottom) {
            const double progress = static_cast<double>(r - (num_cell_rows - K_MERGE_ROWS)) / static_cast<double>(K_MERGE_ROWS);
            p_horizontal = std::max(p_horizontal, 0.50 + 0.45 * (progress * progress)); // Smooth quadratic ramp
        }

        // Shuffle candidate wall order to eliminate left-to-right directional skew
        if (h_candidates.size() > 1 && !is_last_row) {
            for (size_t i = h_candidates.size() - 1; i > 0; --i) {
                const size_t j = static_cast<size_t>(this->re()) % (i + 1);
                std::swap(h_candidates[i], h_candidates[j]);
            }
        }

        for (const size_t c : h_candidates) {
            const int root_A = uf.find(static_cast<int>(c));
            const int root_B = uf.find(static_cast<int>(c + 1));

            if (root_A != root_B) {
                const bool should_merge = is_last_row || (dist_01(this->re) < p_horizontal);

                if (should_merge) {
                    // Carve horizontal wall between room c and room c+1
                    maze[2 * r + 1][2 * c + 2] = PATH;
                    cum_actual_h += 1.0;
                    uf.unite(root_A, root_B);
                }
            }
        }

        // 3. Vertical Carving Phase (for all rows except the last)
        if (!is_last_row) {
            std::fill(incoming_drop_token.begin(), incoming_drop_token.end(), -1);

            // Group columns by their Union-Find root in O(C) using intrusive linked list
            active_roots.clear();
            for (size_t c = 0; c < num_cell_cols; ++c) {
                const int root = uf.find(static_cast<int>(c));
                if (head[root] == -1) {
                    active_roots.push_back(root);
                }
                next_col[c] = head[root];
                head[root] = static_cast<int>(c);
                set_size[root]++;
            }

            for (size_t token = 0; token < active_roots.size(); ++token) {
                const int root = active_roots[token];
                const int k = set_size[root];

                // Extract columns for this set into col_buffer
                col_buffer.clear();
                int curr_c = head[root];
                while (curr_c != -1) {
                    col_buffer.push_back(curr_c);
                    curr_c = next_col[curr_c];
                }

                // Clean up intrusive list state for next row
                head[root] = -1;
                set_size[root] = 0;

                if (k == 1) {
                    // Singleton set: exactly 1 vertical drop to guarantee tree connectivity
                    const size_t c = col_buffer[0];
                    maze[2 * r + 2][2 * c + 1] = PATH;
                    incoming_drop_token[c] = static_cast<int>(token);
                } else {
                    // k >= 2: Calibrated drop distribution to guarantee E[drops | k] = k / 2
                    // Step A: Pick 1 cell uniformly at random to guarantee >= 1 drop
                    const size_t mandatory_idx = static_cast<size_t>(this->re()) % k;
                    const size_t mand_c = col_buffer[mandatory_idx];
                    maze[2 * r + 2][2 * mand_c + 1] = PATH;
                    incoming_drop_token[mand_c] = static_cast<int>(token);

                    // Step B: Each remaining cell drops with calibrated probability
                    // p_extra(k) = (k/2 - 1) / (k - 1), ensuring total expected drops = k / 2
                    const double p_extra = std::max(0.0, std::min(0.5, (static_cast<double>(k) / 2.0 - 1.0) / (static_cast<double>(k) - 1.0)));

                    for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
                        if (i == mandatory_idx) continue;
                        const size_t opt_c = col_buffer[i];

                        if (dist_01(this->re) < p_extra) {
                            maze[2 * r + 2][2 * opt_c + 1] = PATH;
                            incoming_drop_token[opt_c] = static_cast<int>(token);
                        }
                    }
                }
            }
        }
    }
}
