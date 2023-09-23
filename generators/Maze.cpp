#include "Maze.hpp"

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


    switch(randomInRange(0,3)) {
        case 0://north
            start.first = 0;
            start.second = randomOdd(0,width-1);
            break;
        case 1://east
            start.first = randomOdd(0,height-1);
            start.second = width-1;
            break;
        case 2://south
            start.first = height-1;
            start.second = randomOdd(0,width-1);
            break;
        case 3://west
            start.first = randomOdd(0,height-1);
            start.second = 0;
            break;
    }
    maze[start.first][start.second] = PATH;

    switch(randomInRange(0,3)) {
        case 0://north
            end.first = 0;
            end.second = randomOdd(0,width-1);
            break;
        case 1://east
            end.first = randomOdd(0,height-1);
            end.second = width-1;
            break;
        case 2://south
            end.first = height-1;
            end.second = randomOdd(0,width-1);
            break;
        case 3://west
            end.first = randomOdd(0,height-1);
            end.second = 0;
            break;
    }
    maze[end.first][end.second] = PATH;

}

Maze::Maze(int height, int width, unsigned int seed) {

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

Maze::Maze(int height, int width) {

    unsigned int seed = rd();

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

    std::string cell[2][2][2][2][2] = {{{{{"\u001b[30m┼\u001b[0m","\u001b[30m├\u001b[0m"},{"\u001b[30m┴\u001b[0m","\u001b[30m└\u001b[0m"}},{{"\u001b[30m┤\u001b[0m","\u001b[30m│\u001b[0m"},{"\u001b[30m┘\u001b[0m","\u001b[30m╵\u001b[0m"}}},{{{"\u001b[30m┬\u001b[0m","\u001b[30m┌\u001b[0m"},{"\u001b[30m─\u001b[0m","\u001b[30m╶\u001b[0m"}},{{"\u001b[30m┐\u001b[0m","\u001b[30m╷\u001b[0m"},{"\u001b[30m╴\u001b[0m","\u001b[30m·\u001b[0m"}}}},{{{{"\u001b[37m·\u001b[0m","\u001b[37m╴\u001b[0m"},{"\u001b[37m╷\u001b[0m","\u001b[37m┐\u001b[0m"}},{{"\u001b[37m╶\u001b[0m","\u001b[37m─\u001b[0m"},{"\u001b[37m┌\u001b[0m","\u001b[37m┬\u001b[0m"}}},{{{"\u001b[37m╵\u001b[0m","\u001b[37m┘\u001b[0m"},{"\u001b[37m│\u001b[0m","\u001b[37m┤\u001b[0m"}},{{"\u001b[37m└\u001b[0m","\u001b[37m┴\u001b[0m"},{"\u001b[37m├\u001b[0m","\u001b[37m┼\u001b[0m"}}}}};
    bool C, N, E, S, W;
    std::string s = "";

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {

            C = maze[h][w];
            N = (h == 0) ? PATH : maze[h - 1][w];
            E = (w == width - 1) ? PATH : maze[h][w + 1];
            S = (h == height - 1) ? PATH : maze[h + 1][w];
            W = (w == 0) ? PATH : maze[h][w - 1];

            s.append(cell[C][N][E][S][W]);
        }
        s.append("\n");
    }
    return s;
}

std::string Maze::mazeToStringSimple() {
    std::string s = "";

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

void Maze::png () {

    png::image<png::gray_pixel_1> image(this->width, this->height);
    png::palette palette = {png::color(0,0,0), png::color(255,255,255)};
    image.set_palette(palette);

    for (png::uint_32 h = 0; h < image.get_height(); ++h) {
        for (png::uint_32 w = 0; w < image.get_width(); ++w) {
            if (this->maze[h][w]) {
                image[h][w] = png::gray_pixel_1(0);
            } else {
                image[h][w] = png::gray_pixel_1(1);
            }
        }
    }

    std::string fileName = std::to_string(this->seed) + "_" + std::to_string(this->height) + "_" + std::to_string(this->width) + ".png";
    image.write(fileName);

}
