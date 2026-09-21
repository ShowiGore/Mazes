#include "Solver.hpp"
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

std::string Solver::save_solution(const Maze &maze_object, const std::string &custom_filepath) {
    const std::vector<std::vector<bool>> &maze = maze_object.getMaze();

    png::image<png::index_pixel_2> image(this->width, this->height);
    const png::palette palette = {
        png::color(0, 0, 0),       // 0: black (wall)
        png::color(255, 255, 255), // 1: white (unvisited path)
        png::color(255, 0, 0),     // 2: red (visited path)
        png::color(0, 255, 0)      // 3: green (solution path)
    };
    image.set_palette(palette);
    image.set_compression_type(png::compression_type_default);

    for (png::uint_32 h = 0; h < image.get_height(); ++h) {
        for (png::uint_32 w = 0; w < image.get_width(); ++w) {
            if (maze[h][w]) {
                image[h][w] = png::index_pixel_2(0); // black
            } else {
                if (this->solution[h][w]) {
                    image[h][w] = png::index_pixel_2(3); // green
                } else if (this->visited[h][w]) {
                    image[h][w] = png::index_pixel_2(2); // red
                } else {
                    image[h][w] = png::index_pixel_2(1); // white
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

    image.write(fullPath);
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