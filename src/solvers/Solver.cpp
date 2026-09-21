#include "Solver.hpp"
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

std::string Solver::save_solution(const Maze &maze_object, const std::string &custom_filepath) {
    const std::vector<std::vector<bool>> &maze = maze_object.getMaze();

    bool has_pruned_cells = false;
    if (!this->pruned.empty()) {
        #pragma omp parallel for reduction(||:has_pruned_cells) schedule(static)
        for (int r = 0; r < this->height; ++r) {
            if (!has_pruned_cells) {
                for (int c = 0; c < this->width; ++c) {
                    if (this->pruned[r][c]) {
                        has_pruned_cells = true;
                        break;
                    }
                }
            }
        }
    }

    std::string fullPath = custom_filepath;
    if (fullPath.empty()) {
        const std::filesystem::path dir = std::filesystem::path(PROJECT_ROOT_DIR) / "generated_mazes" / "images";
        std::filesystem::create_directories(dir);
        fullPath = (dir / std::format("{}_{}_{}_{}_{}.png",
                                      maze_object.getSeed(),
                                      this->height,
                                      this->width,
                                      maze_object.getGeneratorName(),
                                      this->solver_name)).string();
    }

    // -------------------------------------------------------------------------
    // 1. Compact 4-color PNG (2 bits per pixel, png::index_pixel_2)
    // 2-bit indexed PNG packs 4 pixels per byte, delivering 2x smaller memory footprint
    // and significantly faster compression than 4-bit/8-bit PNGs.
    // Paul Tol's Bright Colorblind-Safe Palette:
    // - 0: Black (0, 0, 0)         -> Wall
    // - 1: White (255, 255, 255)   -> Unexplored path
    // - 2: Tol Red (238, 102, 119) -> Explored or Pruned path (processed by solver)
    // - 3: Tol Green (34, 136, 51) -> Solution path
    // -------------------------------------------------------------------------
    {
        png::image<png::index_pixel_2> image(this->width, this->height);
        const png::palette palette = {
            png::color(0, 0, 0),         // 0: black (wall)
            png::color(255, 255, 255),   // 1: white (unexplored path)
            png::color(238, 102, 119),   // 2: Tol red / rose (explored or pruned)
            png::color(34, 136, 51)      // 3: Tol green (solution path)
        };
        image.set_palette(palette);
        image.set_compression_type(png::compression_type_default);

        for (png::uint_32 h = 0; h < image.get_height(); ++h) {
            for (png::uint_32 w = 0; w < image.get_width(); ++w) {
                if (maze[h][w]) {
                    image[h][w] = png::index_pixel_2(0); // black (wall)
                } else if (this->solution[h][w]) {
                    image[h][w] = png::index_pixel_2(3); // Tol green (solution)
                } else if (this->visited[h][w] || (has_pruned_cells && this->pruned[h][w])) {
                    image[h][w] = png::index_pixel_2(2); // Tol red (explored or pruned)
                } else {
                    image[h][w] = png::index_pixel_2(1); // white (unexplored path)
                }
            }
        }
        image.write(fullPath);
    }

    // -------------------------------------------------------------------------
    // 2. Detailed 5-color PNG (4 bits per pixel, png::index_pixel_4)
    // Generated ONLY when the solver has pruned cells to visually distinguish
    // dead-end pruning (Tol Blue) from active search exploration (Tol Red).
    // Paul Tol's Bright Colorblind-Safe Palette:
    // - 0: Black (0, 0, 0)         -> Wall
    // - 1: White (255, 255, 255)   -> Unexplored path
    // - 2: Tol Red (238, 102, 119) -> Explored path by search
    // - 3: Tol Green (34, 136, 51) -> Solution path
    // - 4: Tol Blue (68, 119, 170) -> Pruned dead ends
    // -------------------------------------------------------------------------
    if (has_pruned_cells) {
        std::filesystem::path p(fullPath);
        std::string detailedPath = (p.parent_path() / (p.stem().string() + "_detailed" + p.extension().string())).string();

        png::image<png::index_pixel_4> image(this->width, this->height);
        const png::palette palette = {
            png::color(0, 0, 0),         // 0: black (wall)
            png::color(255, 255, 255),   // 1: white (unexplored path)
            png::color(238, 102, 119),   // 2: Tol red / rose (explored path by search)
            png::color(34, 136, 51),     // 3: Tol green (solution path)
            png::color(68, 119, 170)     // 4: Tol blue (pruned dead ends)
        };
        image.set_palette(palette);
        image.set_compression_type(png::compression_type_default);

        for (png::uint_32 h = 0; h < image.get_height(); ++h) {
            for (png::uint_32 w = 0; w < image.get_width(); ++w) {
                if (maze[h][w]) {
                    image[h][w] = png::index_pixel_4(0); // black (wall)
                } else if (this->solution[h][w]) {
                    image[h][w] = png::index_pixel_4(3); // Tol green (solution)
                } else if (this->visited[h][w]) {
                    image[h][w] = png::index_pixel_4(2); // Tol red (explored path)
                } else if (this->pruned[h][w]) {
                    image[h][w] = png::index_pixel_4(4); // Tol blue (pruned dead end)
                } else {
                    image[h][w] = png::index_pixel_4(1); // white (unexplored path)
                }
            }
        }
        image.write(detailedPath);
        std::cout << "Saved detailed 5-color solution image: " << detailedPath << std::endl;
    }

    return fullPath;
}

std::string Solver::save_binary_solution(const Maze &maze_object, const std::string &custom_filepath) const {
    std::string fullPath = custom_filepath;
    if (fullPath.empty()) {
        const std::filesystem::path dir = std::filesystem::path(PROJECT_ROOT_DIR) / "generated_mazes" / "binary";
        std::filesystem::create_directories(dir);
        fullPath = (dir / std::format("{}_{}_{}_{}_{}.sol",
                                      maze_object.getSeed(),
                                      this->height,
                                      this->width,
                                      maze_object.getGeneratorName(),
                                      this->solver_name)).string();
    }

    std::ofstream out(fullPath, std::ios::binary);
    if (!out) {
        std::cerr << "Error: cannot open solution file for writing: " << fullPath << std::endl;
        return "";
    }

    // Binary Solution Header:
    // Magic: "SOLU" (4 bytes)
    constexpr char magic[4] = {'S', 'O', 'L', 'U'};
    out.write(magic, 4);

    const uint32_t version = 1;
    const uint32_t seed_val = maze_object.getSeed();
    const uint64_t h = this->height;
    const uint64_t w = this->width;
    const uint32_t solver_len = static_cast<uint32_t>(this->solver_name.size());

    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&seed_val), sizeof(seed_val));
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(&w), sizeof(w));
    out.write(reinterpret_cast<const char*>(&solver_len), sizeof(solver_len));
    out.write(this->solver_name.data(), solver_len);

    // Payload: Bit-packed solution (1 bit per cell: 1 = solution, 0 = not solution)
    const size_t total_bits = static_cast<size_t>(this->height) * this->width;
    const size_t total_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> buffer(total_bytes, 0);

    size_t bit_idx = 0;
    for (int r = 0; r < this->height; ++r) {
        for (int c = 0; c < this->width; ++c) {
            if (this->solution[r][c]) {
                buffer[bit_idx / 8] |= (1 << (bit_idx % 8));
            }
            ++bit_idx;
        }
    }

    out.write(reinterpret_cast<const char*>(buffer.data()), total_bytes);
    return fullPath;
}