#include "Maze.hpp"
#include <filesystem>
#include <format>

int Maze::randomInRange (int min, int max) {
    return min + (rand() % (max-min+1));
}

int Maze::randomEven (int min, int max) {
    if (max % 2 != 0) --max;
    if (min % 2 != 0) ++min;
    //return min + 2 * (random.nextInt((max-min)/2+1));
    return min + 2 * (rand() % ((max-min)/2+1));
}

int Maze::randomOdd (int min, int max) {
    if (max % 2 == 0) --max;
    if (min % 2 == 0) ++min;
    //return min + 2 * (random.nextInt((max-min)/2+1));
    return min + 2 * (rand() % ((max-min)/2+1));
}

void Maze::init() {
    int aux;
    for (int h = 0; h < this->height; h++) {
        for (int w = 0; w < this->width; w++) {
            if ( h==0 || h==(this->height-1) || w==0 || w==(this->width-1) || !( (h&1) || (w&1) ) ){
                maze[h][w] = true;
            } else {
                maze[h][w] = false;
            }
        }
    }
}

void Maze::buildStartEnd() {

    std::vector<std::pair<int, int>> perimeter;
    perimeter.reserve(2 * (width / 2) + 2 * (height / 2));

    for (int w = 1; w < width - 1; w += 2) {
        perimeter.emplace_back(0, w);          // North
        perimeter.emplace_back(height - 1, w); // South
    }
    for (int h = 1; h < height - 1; h += 2) {
        perimeter.emplace_back(h, 0);         // West
        perimeter.emplace_back(h, width - 1); // East
    }

    if (perimeter.size() < 2) return; // Small mazes

    const int size = static_cast<int>(perimeter.size());
    const int idx1 = randomInRange(0, size - 1);
    int idx2 = randomInRange(0, size - 2);
    if (idx2 >= idx1) idx2++;

    start = perimeter[idx1];
    maze[start.first][start.second] = PATH;
    end = perimeter[idx2];
    maze[end.first][end.second] = PATH;

}

Maze::Maze(const int height, const int width, const unsigned int seed) {

    this->height = (height < 3) ? (3) : ((height & 1) ? height : (height+1));
    this->width = (width < 3) ? (3) : ((width & 1) ? width : (width+1));
    this->seed = seed;
    this->re.seed(seed);
    srand(seed);

    this->maze.resize(this->height, std::vector<bool>(this->width));
    this->visited.resize(this->height, std::vector<bool>(this->width));
    this->solution.resize(this->height, std::vector<bool>(this->width));

    std::cout << seed << "[" << this->height << "]" << "[" << this->width << "]" << std::endl;
}

Maze::Maze(const int height, const int width) {

    const unsigned int seed = rd();

    this->height = (height < 3) ? (3) : ((height & 1) ? height : (height+1));
    this->width = (width < 3) ? (3) : ((width & 1) ? width : (width+1));
    this->seed = seed;
    this->re.seed(seed);
    srand(seed);

    this->maze.resize(this->height, std::vector<bool>(this->width));
    this->visited.resize(this->height, std::vector<bool>(this->width));
    this->solution.resize(this->height, std::vector<bool>(this->width));

    std::cout << seed << "[" << this->height << "]" << "[" << this->width << "]" << std::endl;

}

void Maze::set(int h, int w, bool b) {
    maze[h][w] = b;
}

bool Maze::get(int h, int w) {
    return maze[h][w];
}

std::string Maze::mazeToString() {
    std::string s;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const std::string cell[2][2][2][2][2] = {{{{{"\u001b[30m┼\u001b[0m","\u001b[30m├\u001b[0m"},{"\u001b[30m┴\u001b[0m","\u001b[30m└\u001b[0m"}},{{"\u001b[30m┤\u001b[0m","\u001b[30m│\u001b[0m"},{"\u001b[30m┘\u001b[0m","\u001b[30m╵\u001b[0m"}}},{{{"\u001b[30m┬\u001b[0m","\u001b[30m┌\u001b[0m"},{"\u001b[30m─\u001b[0m","\u001b[30m╶\u001b[0m"}},{{"\u001b[30m┐\u001b[0m","\u001b[30m╷\u001b[0m"},{"\u001b[30m╴\u001b[0m","\u001b[30m·\u001b[0m"}}}},{{{{"\u001b[37m·\u001b[0m","\u001b[37m╴\u001b[0m"},{"\u001b[37m╷\u001b[0m","\u001b[37m┐\u001b[0m"}},{{"\u001b[37m╶\u001b[0m","\u001b[37m─\u001b[0m"},{"\u001b[37m┌\u001b[0m","\u001b[37m┬\u001b[0m"}}},{{{"\u001b[37m╵\u001b[0m","\u001b[37m┘\u001b[0m"},{"\u001b[37m│\u001b[0m","\u001b[37m┤\u001b[0m"}},{{"\u001b[37m└\u001b[0m","\u001b[37m┴\u001b[0m"},{"\u001b[37m├\u001b[0m","\u001b[37m┼\u001b[0m"}}}}};

            const bool C = maze[h][w];
            const bool N = (h == 0) ? PATH : maze[h - 1][w];
            const bool E = (w == width - 1) ? PATH : maze[h][w + 1];
            const bool S = (h == height - 1) ? PATH : maze[h + 1][w];
            const bool W = (w == 0) ? PATH : maze[h][w - 1];

            s.append(cell[C][N][E][S][W]);
        }
        s.append("\n");
    }
    return s;
}

std::string Maze::mazeToStringSimple() {
    std::string s;

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            if (maze[h][w] == WALL) {
                s.append("\u001b[37m█\u001b[0m");
            } else {
                s.append("\u001b[30m█\u001b[0m");
            }
        }
        s.append("\n");
    }

    return s;
}

void Maze::print () { std::cout << mazeToString(); }

void Maze::printSimple () { std::cout << mazeToStringSimple(); }

void Maze::save_maze () {

    png::image<png::gray_pixel_1> image(this->width, this->height);
    image.set_compression_type(png::compression_type_default);

    for (png::uint_32 h = 0; h < image.get_height(); ++h) {
        for (png::uint_32 w = 0; w < image.get_width(); ++w) {
            if (this->maze[h][w]) {
                image[h][w] = png::gray_pixel_1(0);
            } else {
                image[h][w] = png::gray_pixel_1(1);
            }
        }
    }

    const std::filesystem::path dir = std::filesystem::path(PROJECT_ROOT_DIR) / "generated_mazes";
    std::filesystem::create_directories(dir);
    const std::string fullPath = (dir / std::format("{}_{}_{}_maze.png", seed, height, width)).string();

    image.write(fullPath);

}

void Maze::save_solution () {

    png::image<png::index_pixel_2> image(this->width, this->height);
    const png::palette palette = {png::color(0,0,0), png::color(255,255,255), png::color(255,0,0), png::color(0,255,0)};  //{black, white, red, green}
    image.set_palette(palette);
    image.set_compression_type(png::compression_type_default);

    for (png::uint_32 h = 0; h < image.get_height(); ++h) {
        for (png::uint_32 w = 0; w < image.get_width(); ++w) {

            if (this->maze[h][w]) { //wall
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
    const std::string fullPath = (dir / std::format("{}_{}_{}_solution.png", seed, height, width)).string();
    image.write(fullPath);

}



void Maze::solve() { //backtracking
    constexpr Direction directions[] = {UP, RIGHT, DOWN, LEFT};
    std::stack<Direction> steps;

    std::pair <int, int> current = start;
    this->visited[current.first][current.second] = true;

    int lastStep = -1;


    //Search end
    while (current != this->end && !(lastStep == N_DIRECTIONS-1 && current == this->start)) {

        bool stepped = false;
        int d = lastStep+1;
        while (d <= N_DIRECTIONS && !stepped) {
            if (d == N_DIRECTIONS) { //backtrack
                lastStep = steps.top();
                steps.pop();
                switch (lastStep) {
                    case UP:
                        current.first = current.first+1;
                        //std::cout << "go back DOWN" << std::endl;
                        break;
                    case RIGHT:
                        current.second = current.second-1;
                        //std::cout << "go back LEFT" << std::endl;
                        break;
                    case DOWN:
                        current.first = current.first-1;
                        //std::cout << "go back UP" << std::endl;
                        break;
                    case LEFT:
                        current.second = current.second+1;
                        //std::cout << "go back RIGHT" << std::endl;
                        break;
                }
            } else {
                switch (directions[d]) {
                    case UP:
                        if (current.first > 0 && this->maze[current.first-1][current.second] == PATH && !this->visited[current.first-1][current.second]) {
                            steps.push(UP);
                            current.first = current.first-1;
                            this->visited[current.first][current.second] = true;
                            lastStep = -1;
                            stepped = true;
                            //std::cout << "go UP" << std::endl;
                        }
                        break;
                    case RIGHT:
                        if (current.second < this->width-1 && this->maze[current.first][current.second+1] == PATH && !this->visited[current.first][current.second+1]) {
                            steps.push(RIGHT);
                            current.second = current.second+1;
                            this->visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go RIGHT" << std::endl;
                        }
                        break;
                    case DOWN:
                        if (current.first < this->height-1 && this->maze[current.first+1][current.second] == PATH && !this->visited[current.first+1][current.second]) {
                            steps.push(DOWN);
                            current.first = current.first+1;
                            this->visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go DOWN" << std::endl;
                        }
                        break;
                    case LEFT:
                        if (current.second > 0 && this->maze[current.first][current.second-1] == PATH && !this->visited[current.first][current.second-1]) {
                            steps.push(LEFT);
                            current.second = current.second-1;
                            this->visited[current.first][current.second] = true;
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
    this->solution[current.first][current.second] = true;
    while (!steps.empty()) {
        lastStep = steps.top();
        steps.pop();
        switch (lastStep) {
            case UP:
                current.first = current.first+1;
                break;
            case RIGHT:
                current.second = current.second-1;
                break;
            case DOWN:
                current.first = current.first-1;
                break;
            case LEFT:
                current.second = current.second+1;
                break;
        }
        this->solution[current.first][current.second] = true;
    }



}
