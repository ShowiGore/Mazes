#include "Solver.hpp"
#include <filesystem>
#include <format>

void Solver::save_solution (const Maze &maze_object) {

    const std::vector<std::vector<bool>> &maze = maze_object.getMaze();

    png::image<png::index_pixel_2> image(this->width, this->height);
    const png::palette palette = {png::color(0,0,0), png::color(255,255,255), png::color(255,0,0), png::color(0,255,0)};  //{black, white, red, green}
    image.set_palette(palette);
    image.set_compression_type(png::compression_type_default);

    for (png::uint_32 h = 0; h < image.get_height(); ++h) {
        for (png::uint_32 w = 0; w < image.get_width(); ++w) {

            if (maze[h][w]) { //wall
                image[h][w] = png::index_pixel_2(0);        //black
            } else {
                if (this->solution[h][w]) {
                    image[h][w] = png::index_pixel_2(3);    //green
                } else if (this->visited[h][w]) {
                    image[h][w] = png::index_pixel_2(2);    //red
                } else { //unvisited path
                    image[h][w] = png::index_pixel_2(1);    //white
                }

            }

        }
    }

    const std::filesystem::path dir = std::filesystem::path(PROJECT_ROOT_DIR) / "generated_mazes";
    std::filesystem::create_directories(dir);
    const std::string fullPath = (dir / std::format("{}_{}_{}_solution.png", maze_object.getSeed(), height, width)).string();
    image.write(fullPath);

}