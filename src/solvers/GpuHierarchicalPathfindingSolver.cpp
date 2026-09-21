#include "GpuHierarchicalPathfindingSolver.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <unordered_map>

#include "utilities/GpuUtils.hpp"

static const char* EMBEDDED_HPA_KERNELS = R"(
#define TILE_DIM 32

__kernel void hpa_find_horizontal_portals(
    __global const int* state_grid,
    const int height,
    const int width,
    const int num_tile_rows,
    const int num_tile_cols,
    __global int* portal_cell_A,
    __global int* portal_cell_B,
    __global int* portal_tile_A,
    __global int* portal_tile_B,
    __global int* portal_counter,
    const int max_portals
) {
    const int tc = get_global_id(0);
    const int tr = get_global_id(1);

    if (tr >= num_tile_rows - 1 || tc >= num_tile_cols) return;

    const int r_A = (tr + 1) * TILE_DIM - 1;
    const int r_B = (tr + 1) * TILE_DIM;

    if (r_B >= height) return;

    const int c_start = tc * TILE_DIM;
    const int c_end = min((tc + 1) * TILE_DIM, width);

    for (int c = c_start; c < c_end; ++c) {
        const int idx_A = r_A * width + c;
        const int idx_B = r_B * width + c;

        if (state_grid[idx_A] != 0xFF && state_grid[idx_B] != 0xFF) {
            const int pid = atomic_inc(portal_counter);
            if (pid < max_portals) {
                portal_cell_A[pid] = idx_A;
                portal_cell_B[pid] = idx_B;
                portal_tile_A[pid] = tr * num_tile_cols + tc;
                portal_tile_B[pid] = (tr + 1) * num_tile_cols + tc;
            }
        }
    }
}

__kernel void hpa_find_vertical_portals(
    __global const int* state_grid,
    const int height,
    const int width,
    const int num_tile_rows,
    const int num_tile_cols,
    __global int* portal_cell_A,
    __global int* portal_cell_B,
    __global int* portal_tile_A,
    __global int* portal_tile_B,
    __global int* portal_counter,
    const int max_portals
) {
    const int tc = get_global_id(0);
    const int tr = get_global_id(1);

    if (tr >= num_tile_rows || tc >= num_tile_cols - 1) return;

    const int c_A = (tc + 1) * TILE_DIM - 1;
    const int c_B = (tc + 1) * TILE_DIM;

    if (c_B >= width) return;

    const int r_start = tr * TILE_DIM;
    const int r_end = min((tr + 1) * TILE_DIM, height);

    for (int r = r_start; r < r_end; ++r) {
        const int idx_A = r * width + c_A;
        const int idx_B = r * width + c_B;

        if (state_grid[idx_A] != 0xFF && state_grid[idx_B] != 0xFF) {
            const int pid = atomic_inc(portal_counter);
            if (pid < max_portals) {
                portal_cell_A[pid] = idx_A;
                portal_cell_B[pid] = idx_B;
                portal_tile_A[pid] = tr * num_tile_cols + tc;
                portal_tile_B[pid] = tr * num_tile_cols + (tc + 1);
            }
        }
    }
}
)";

struct GpuHierarchicalPathfindingSolver::Impl {
    GpuContext gpu;
    cl::Program program;
    cl::Kernel kernel_horiz;
    cl::Kernel kernel_vert;
};

GpuHierarchicalPathfindingSolver::GpuHierarchicalPathfindingSolver(std::string device_vendor)
    : GpuSolver(std::move(device_vendor)),
      pimpl(std::make_unique<Impl>()) {
    this->solver_name = "gpu-hpa";
}

GpuHierarchicalPathfindingSolver::~GpuHierarchicalPathfindingSolver() = default;
GpuHierarchicalPathfindingSolver::GpuHierarchicalPathfindingSolver(GpuHierarchicalPathfindingSolver&&) noexcept = default;
GpuHierarchicalPathfindingSolver& GpuHierarchicalPathfindingSolver::operator=(GpuHierarchicalPathfindingSolver&&) noexcept = default;

bool GpuHierarchicalPathfindingSolver::ensureOpenCLInitialized() {
    if (pimpl->gpu.ready) return true;

    if (!pimpl->gpu.init(this->preferred_device_vendor, "[GpuHierarchicalPathfindingSolver]")) {
        return false;
    }
    this->selected_platform_name = pimpl->gpu.platform_name;
    this->selected_device_name = pimpl->gpu.device_name;

    const std::string kernel_source = GpuContext::loadKernelSource("hpa_kernels.cl", EMBEDDED_HPA_KERNELS);
    pimpl->program = pimpl->gpu.buildProgram(kernel_source, "-cl-std=CL2.0 -cl-fast-relaxed-math", "[GpuHierarchicalPathfindingSolver]");
    if (!pimpl->gpu.ready) return false;

    pimpl->kernel_horiz = cl::Kernel(pimpl->program, "hpa_find_horizontal_portals");
    pimpl->kernel_vert = cl::Kernel(pimpl->program, "hpa_find_vertical_portals");
    return true;
}

bool GpuHierarchicalPathfindingSolver::solve(const Maze &maze_object) {
    if (!ensureOpenCLInitialized()) {
        std::cerr << "[GpuHierarchicalPathfindingSolver] Failed to initialize OpenCL runtime.\n";
        return false;
    }

    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();
    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &grid = maze_object.getMaze();
    const size_t total_cells = static_cast<size_t>(this->height) * this->width;

    std::vector<int> host_state(total_cells);
    for (int r = 0; r < this->height; ++r) {
        const size_t row_offset = static_cast<size_t>(r) * this->width;
        for (int c = 0; c < this->width; ++c) {
            host_state[row_offset + c] = grid[r][c] ? 0xFF : 0x00;
        }
    }

    constexpr int TILE_DIM = 32;
    const int num_tile_rows = (this->height + TILE_DIM - 1) / TILE_DIM;
    const int num_tile_cols = (this->width + TILE_DIM - 1) / TILE_DIM;
    const int max_portals = 2000000;

    try {
        cl::Buffer buf_state(pimpl->gpu.context, CL_MEM_READ_ONLY, total_cells * sizeof(int));
        pimpl->gpu.queue.enqueueWriteBuffer(buf_state, CL_TRUE, 0, total_cells * sizeof(int), host_state.data());

        cl::Buffer buf_portal_cell_A(pimpl->gpu.context, CL_MEM_READ_WRITE, max_portals * sizeof(int));
        cl::Buffer buf_portal_cell_B(pimpl->gpu.context, CL_MEM_READ_WRITE, max_portals * sizeof(int));
        cl::Buffer buf_portal_tile_A(pimpl->gpu.context, CL_MEM_READ_WRITE, max_portals * sizeof(int));
        cl::Buffer buf_portal_tile_B(pimpl->gpu.context, CL_MEM_READ_WRITE, max_portals * sizeof(int));
        cl::Buffer buf_portal_counter(pimpl->gpu.context, CL_MEM_READ_WRITE, sizeof(int));

        const int zero = 0;
        pimpl->gpu.queue.enqueueWriteBuffer(buf_portal_counter, CL_TRUE, 0, sizeof(int), &zero);

        // 1. Launch Horizontal Portals Kernel
        pimpl->kernel_horiz.setArg(0, buf_state);
        pimpl->kernel_horiz.setArg(1, this->height);
        pimpl->kernel_horiz.setArg(2, this->width);
        pimpl->kernel_horiz.setArg(3, num_tile_rows);
        pimpl->kernel_horiz.setArg(4, num_tile_cols);
        pimpl->kernel_horiz.setArg(5, buf_portal_cell_A);
        pimpl->kernel_horiz.setArg(6, buf_portal_cell_B);
        pimpl->kernel_horiz.setArg(7, buf_portal_tile_A);
        pimpl->kernel_horiz.setArg(8, buf_portal_tile_B);
        pimpl->kernel_horiz.setArg(9, buf_portal_counter);
        pimpl->kernel_horiz.setArg(10, max_portals);

        const cl::NDRange global_tiles(
            ((num_tile_cols + 15) / 16) * 16,
            ((num_tile_rows + 15) / 16) * 16
        );
        const cl::NDRange local_tiles(16, 16);

        pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_horiz, cl::NullRange, global_tiles, local_tiles);

        // 2. Launch Vertical Portals Kernel
        pimpl->kernel_vert.setArg(0, buf_state);
        pimpl->kernel_vert.setArg(1, this->height);
        pimpl->kernel_vert.setArg(2, this->width);
        pimpl->kernel_vert.setArg(3, num_tile_rows);
        pimpl->kernel_vert.setArg(4, num_tile_cols);
        pimpl->kernel_vert.setArg(5, buf_portal_cell_A);
        pimpl->kernel_vert.setArg(6, buf_portal_cell_B);
        pimpl->kernel_vert.setArg(7, buf_portal_tile_A);
        pimpl->kernel_vert.setArg(8, buf_portal_tile_B);
        pimpl->kernel_vert.setArg(9, buf_portal_counter);
        pimpl->kernel_vert.setArg(10, max_portals);

        pimpl->gpu.queue.enqueueNDRangeKernel(pimpl->kernel_vert, cl::NullRange, global_tiles, local_tiles);

        int portal_count = 0;
        pimpl->gpu.queue.enqueueReadBuffer(buf_portal_counter, CL_TRUE, 0, sizeof(int), &portal_count);
        portal_count = std::min(portal_count, max_portals);
        total_portals = portal_count;

        std::vector<int> p_cell_A(portal_count);
        std::vector<int> p_cell_B(portal_count);
        std::vector<int> p_tile_A(portal_count);
        std::vector<int> p_tile_B(portal_count);

        if (portal_count > 0) {
            pimpl->gpu.queue.enqueueReadBuffer(buf_portal_cell_A, CL_FALSE, 0, portal_count * sizeof(int), p_cell_A.data());
            pimpl->gpu.queue.enqueueReadBuffer(buf_portal_cell_B, CL_FALSE, 0, portal_count * sizeof(int), p_cell_B.data());
            pimpl->gpu.queue.enqueueReadBuffer(buf_portal_tile_A, CL_FALSE, 0, portal_count * sizeof(int), p_tile_A.data());
            pimpl->gpu.queue.enqueueReadBuffer(buf_portal_tile_B, CL_TRUE, 0, portal_count * sizeof(int), p_tile_B.data());
        }

        std::cout << "[GpuHierarchicalPathfindingSolver] Extracted " << portal_count
                  << " portals across " << (num_tile_rows * num_tile_cols) << " tiles on GPU\n";

        // 3. Construct Abstract Graph on Host:
        // Nodes: Portals [0 .. portal_count - 1], Start = portal_count, End = portal_count + 1
        const int START_NODE = portal_count;
        const int END_NODE = portal_count + 1;
        const int TOTAL_NODES = portal_count + 2;

        struct Edge {
            int to;
            int cost;
            int from_cell;
            int to_cell;
        };

        std::vector<std::vector<Edge>> adj(TOTAL_NODES);

        // Inter-tile portal edges (connecting cell_A in tile_A and cell_B in tile_B):
        // Each portal corresponds to 2 endpoints: cell_A and cell_B.
        // Group portals by tile:
        std::vector<std::vector<std::pair<int, int>>> tile_portals(num_tile_rows * num_tile_cols);
        for (int i = 0; i < portal_count; ++i) {
            tile_portals[p_tile_A[i]].push_back({i, p_cell_A[i]});
            tile_portals[p_tile_B[i]].push_back({i, p_cell_B[i]});
        }

        // Add Start and End to their tiles
        const int start_tile = (this->start.first / TILE_DIM) * num_tile_cols + (this->start.second / TILE_DIM);
        const int end_tile = (this->end.first / TILE_DIM) * num_tile_cols + (this->end.second / TILE_DIM);
        const int start_idx = this->start.first * this->width + this->start.second;
        const int end_idx = this->end.first * this->width + this->end.second;

        tile_portals[start_tile].push_back({START_NODE, start_idx});
        tile_portals[end_tile].push_back({END_NODE, end_idx});

        // Intra-tile edges: local BFS within each tile that has >= 2 endpoints
        const int dr[4] = {-1, 0, 1, 0};
        const int dc[4] = {0, 1, 0, -1};

        std::vector<int> local_dist(TILE_DIM * TILE_DIM, -1);
        std::vector<int> q;
        q.reserve(TILE_DIM * TILE_DIM);

        for (int t = 0; t < num_tile_rows * num_tile_cols; ++t) {
            const auto &endpoints = tile_portals[t];
            if (endpoints.size() < 2) continue;

            const int tr = t / num_tile_cols;
            const int tc = t % num_tile_cols;
            const int r_base = tr * TILE_DIM;
            const int c_base = tc * TILE_DIM;

            // Connect endpoints inside this tile
            for (size_t i = 0; i < endpoints.size(); ++i) {
                const auto [u_node, u_cell] = endpoints[i];
                const int u_lr = (u_cell / this->width) - r_base;
                const int u_lc = (u_cell % this->width) - c_base;

                // BFS inside tile from u
                std::fill(local_dist.begin(), local_dist.end(), -1);
                q.clear();

                const int u_lidx = u_lr * TILE_DIM + u_lc;
                local_dist[u_lidx] = 0;
                q.push_back(u_lidx);
                size_t head = 0;

                while (head < q.size()) {
                    const int curr = q[head++];
                    const int cr = curr / TILE_DIM;
                    const int cc = curr % TILE_DIM;
                    const int d = local_dist[curr];

                    for (int k = 0; k < 4; ++k) {
                        const int nr = cr + dr[k];
                        const int nc = cc + dc[k];
                        if (nr < 0 || nr >= TILE_DIM || nc < 0 || nc >= TILE_DIM) continue;
                        const int gr = r_base + nr;
                        const int gc = c_base + nc;
                        if (gr >= this->height || gc >= this->width) continue;
                        if (host_state[gr * this->width + gc] == 0xFF) continue;

                        const int nl = nr * TILE_DIM + nc;
                        if (local_dist[nl] == -1) {
                            local_dist[nl] = d + 1;
                            q.push_back(nl);
                        }
                    }
                }

                // Check reachability to other endpoints in the same tile
                for (size_t j = i + 1; j < endpoints.size(); ++j) {
                    const auto [v_node, v_cell] = endpoints[j];
                    const int v_lr = (v_cell / this->width) - r_base;
                    const int v_lc = (v_cell % this->width) - c_base;
                    const int v_lidx = v_lr * TILE_DIM + v_lc;
                    const int dist = local_dist[v_lidx];

                    if (dist != -1) {
                        adj[u_node].push_back({v_node, dist, u_cell, v_cell});
                        adj[v_node].push_back({u_node, dist, v_cell, u_cell});
                    }
                }
            }
        }

        // 4. Solve Abstract Graph using Bidirectional BFS
        std::vector<int> parent_node(TOTAL_NODES, -1);
        std::vector<Edge> parent_edge(TOTAL_NODES);
        std::vector<int> dist_f(TOTAL_NODES, -1);
        std::vector<int> dist_b(TOTAL_NODES, -1);

        std::queue<int> q_f, q_b;
        q_f.push(START_NODE);
        dist_f[START_NODE] = 0;

        q_b.push(END_NODE);
        dist_b[END_NODE] = 0;

        int meet_u = -1, meet_v = -1;
        Edge meet_edge;

        while (!q_f.empty() && !q_b.empty()) {
            macro_steps++;

            // Forward step
            if (!q_f.empty()) {
                const int u = q_f.front();
                q_f.pop();

                for (const auto &e : adj[u]) {
                    if (dist_b[e.to] != -1) {
                        meet_u = u;
                        meet_v = e.to;
                        meet_edge = e;
                        break;
                    }
                    if (dist_f[e.to] == -1) {
                        dist_f[e.to] = dist_f[u] + e.cost;
                        parent_node[e.to] = u;
                        parent_edge[e.to] = e;
                        q_f.push(e.to);
                    }
                }
                if (meet_u != -1) break;
            }

            // Backward step
            if (!q_b.empty()) {
                const int u = q_b.front();
                q_b.pop();

                for (const auto &e : adj[u]) {
                    if (dist_f[e.to] != -1) {
                        meet_u = e.to;
                        meet_v = u;
                        meet_edge = e;
                        break;
                    }
                    if (dist_b[e.to] == -1) {
                        dist_b[e.to] = dist_b[u] + e.cost;
                        parent_node[e.to] = u;
                        parent_edge[e.to] = e;
                        q_b.push(e.to);
                    }
                }
                if (meet_u != -1) break;
            }
        }

        if (meet_u == -1) {
            std::cerr << "[GpuHierarchicalPathfindingSolver] No macro-path found.\n";
            return false;
        }

        std::cout << "[GpuHierarchicalPathfindingSolver] Macro-path found in "
                  << macro_steps << " abstract graph steps!\n";

        // 5. Reconstruct the full path
        this->solution.assign(this->height, std::vector<bool>(this->width, false));
        this->visited.assign(this->height, std::vector<bool>(this->width, false));

        std::vector<std::pair<int, int>> path_segments;
        path_segments.push_back({meet_edge.from_cell, meet_edge.to_cell});

        // Trace u back to START_NODE
        int curr = meet_u;
        while (curr != START_NODE) {
            path_segments.push_back({parent_edge[curr].from_cell, parent_edge[curr].to_cell});
            curr = parent_node[curr];
        }

        // Trace v back to END_NODE
        curr = meet_v;
        while (curr != END_NODE) {
            path_segments.push_back({parent_edge[curr].from_cell, parent_edge[curr].to_cell});
            curr = parent_node[curr];
        }

        // Reconstruct intra-tile paths for all segments
        for (const auto &[from_c, to_c] : path_segments) {
            const int r1 = from_c / this->width;
            const int c1 = from_c % this->width;
            const int r2 = to_c / this->width;
            const int c2 = to_c % this->width;

            // If adjacent cells across portal
            if (abs(r1 - r2) + abs(c1 - c2) == 1) {
                this->solution[r1][c1] = true;
                this->solution[r2][c2] = true;
                continue;
            }

            // Intra-tile path: local BFS inside tile
            const int tr = r1 / TILE_DIM;
            const int tc = c1 / TILE_DIM;
            const int r_base = tr * TILE_DIM;
            const int c_base = tc * TILE_DIM;

            std::vector<int> l_parent(TILE_DIM * TILE_DIM, -1);
            q.clear();

            const int start_l = (r1 - r_base) * TILE_DIM + (c1 - c_base);
            const int target_l = (r2 - r_base) * TILE_DIM + (c2 - c_base);

            l_parent[start_l] = start_l;
            q.push_back(start_l);
            size_t head = 0;

            while (head < q.size()) {
                const int cur = q[head++];
                if (cur == target_l) break;

                const int cr = cur / TILE_DIM;
                const int cc = cur % TILE_DIM;

                for (int k = 0; k < 4; ++k) {
                    const int nr = cr + dr[k];
                    const int nc = cc + dc[k];
                    if (nr < 0 || nr >= TILE_DIM || nc < 0 || nc >= TILE_DIM) continue;
                    const int gr = r_base + nr;
                    const int gc = c_base + nc;
                    if (gr >= this->height || gc >= this->width) continue;
                    if (host_state[gr * this->width + gc] == 0xFF) continue;

                    const int nl = nr * TILE_DIM + nc;
                    if (l_parent[nl] == -1) {
                        l_parent[nl] = cur;
                        q.push_back(nl);
                    }
                }
            }

            int p_curr = target_l;
            while (p_curr != start_l) {
                const int pr = r_base + (p_curr / TILE_DIM);
                const int pc = c_base + (p_curr % TILE_DIM);
                this->solution[pr][pc] = true;
                p_curr = l_parent[p_curr];
            }
            this->solution[r1][c1] = true;
        }

        return true;

    } catch (const cl::Error &err) {
        std::cerr << "[GpuHierarchicalPathfindingSolver] OpenCL error: " << err.what()
                  << " (" << err.err() << ")\n";
        return false;
    }
}
