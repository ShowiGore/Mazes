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

    // State arrays for vertical drops:
    // incoming_drop_token[c]: token from row r-1 (>= 0 if received drop, -1 otherwise)
    // outgoing_drop_token[c]: token for row r+1 (carved drops from row r)
    std::vector<int> incoming_drop_token(num_cell_cols, -1);
    std::vector<int> outgoing_drop_token(num_cell_cols, -1);
    std::vector<int> drop_leader(num_cell_cols, -1);

    // Candidate horizontal indices for randomized 1D Kruskal pass
    std::vector<size_t> h_candidates(num_cell_cols > 1 ? num_cell_cols - 1 : 0);
    std::iota(h_candidates.begin(), h_candidates.end(), 0);

    // Fast PRNG helper for double in [0, 1)
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);

    // =========================================================================
    // DYNAMIC NON-HOMOGENEOUS ELLER WITH TRANSFER-MATRIX BOUNDARY CORRECTIONS
    // =========================================================================

    for (size_t r = 0; r < num_cell_rows; ++r) {
        const bool is_last_row = (r == num_cell_rows - 1);
        const double tau = (num_cell_rows > 1) ? static_cast<double>(r) / static_cast<double>(num_cell_rows - 1) : 1.0;

        // Reset Union-Find for this row
        uf.reset(num_cell_cols);

        // 1. Carve room paths and connect columns that dropped from the SAME set in row r-1
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

        // 2. Dynamic Horizontal Carving Phase
        // In the bulk, p_h = 0.55 - 0.60 to maintain long, winding horizontal corridors.
        // Near the bottom, p_h smoothly ramps to 1.0.
        double p_horizontal = 0.55;
        if (is_last_row) {
            p_horizontal = 1.0;
        } else if (tau > 0.70) {
            const double ramp = (tau - 0.70) / 0.30;
            p_horizontal = 0.55 + 0.45 * (ramp * ramp); // Smooth quadratic ramp to 1.0
        }

        // Shuffle candidate wall order (Fisher-Yates) to eliminate left-to-right skew
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
                    maze[2 * r + 1][2 * c + 2] = PATH; // Carve horizontal connection
                    uf.unite(root_A, root_B);
                }
            }
        }

        // 3. Dynamic Vertical Carving Phase (for all rows except the last)
        if (!is_last_row) {
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

            // Clear outgoing drop state for row r+1 (DO NOT clear incoming_drop_token!)
            std::fill(outgoing_drop_token.begin(), outgoing_drop_token.end(), -1);

            for (size_t token = 0; token < active_roots.size(); ++token) {
                const int root = active_roots[token];
                const int k = set_size[root];

                // Extract and sort columns for this set
                col_buffer.clear();
                int curr_c = head[root];
                while (curr_c != -1) {
                    col_buffer.push_back(curr_c);
                    curr_c = next_col[curr_c];
                }
                std::sort(col_buffer.begin(), col_buffer.end());

                // Reset intrusive list for next row
                head[root] = -1;
                set_size[root] = 0;

                // Check whether this set has an incoming connection from row r-1
                bool has_incoming_connection = false;
                if (r == 0) {
                    has_incoming_connection = false;
                } else {
                    for (const int col : col_buffer) {
                        if (incoming_drop_token[col] >= 0) {
                            has_incoming_connection = true;
                            break;
                        }
                    }
                }

                // Every disjoint set MUST drop at least 1 cell to row r+1 to guarantee
                // that all components can merge in subsequent rows (topological spanning tree necessity).
                // Step A: Mandatory drop (1 per set, chosen uniformly at random)
                const size_t mand_idx = static_cast<size_t>(this->re()) % k;
                const size_t mand_c = col_buffer[mand_idx];
                maze[2 * r + 2][2 * mand_c + 1] = PATH;
                outgoing_drop_token[mand_c] = static_cast<int>(token);

                // Step B: Optional extra drops for sets of size k >= 2 with Burton-Pemantle Dipole Repulsion
                if (k >= 2) {
                    const double p_extra_base = std::max(0.02, 0.15 * (1.0 - 0.50 * tau));
                    std::vector<size_t> dropped_cols = {mand_c};

                    for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
                        if (i == mand_idx) continue;
                        const size_t opt_c = col_buffer[i];

                        // Compute dipole potential from already dropped columns
                        double potential = 0.0;
                        for (const size_t d : dropped_cols) {
                            const double dist = std::abs(static_cast<double>(opt_c) - static_cast<double>(d));
                            if (dist < 1e-6) continue;
                            potential += 1.0 / (dist * dist);
                        }

                        const double p_eff = p_extra_base * std::exp(-1.5 * potential);

                        if (dist_01(this->re) < p_eff) {
                            maze[2 * r + 2][2 * opt_c + 1] = PATH;
                            outgoing_drop_token[opt_c] = static_cast<int>(token);
                            dropped_cols.push_back(opt_c);
                        }
                    }
                }
            }

            // Transfer outgoing drops to incoming drops for row r+1
            incoming_drop_token = outgoing_drop_token;
        }
    }
}
