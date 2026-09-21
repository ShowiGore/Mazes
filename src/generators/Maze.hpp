#ifndef MAZES_MAZE_HPP
#define MAZES_MAZE_HPP

#include <vector>
#include <array>
#include <stack>
#include <random>
#include <utility>
#include <iostream>
#include <string>
#include <png.hpp>

constexpr bool WALL = true;
constexpr bool PATH = false;
enum Direction {UP=0, RIGHT=1, DOWN=2, LEFT=3};

constexpr int N_DIRECTIONS = 4;

class Maze {

protected:
    std::vector<std::vector<bool>> maze;
    std::pair<int, int> start, end;
    int height = 0, width = 0;
    unsigned int seed = 0;
    std::string generator_name = "unknown";

    std::random_device rd;
    std::mt19937 re{};
    std::bernoulli_distribution bd;

    static int randomInRange(int min, int max);
    static int randomEven(int min, int max);
    static int randomOdd(int min, int max);
    virtual void init();
    void buildStartEnd();
    virtual void generate() {};

public:

    virtual ~Maze() = default;
    Maze() = default;
    Maze(int height, int width, unsigned int seed);
    Maze(int height, int width);

    [[nodiscard]] int getSeed() const;
    [[nodiscard]] int getHeight() const;
    [[nodiscard]] int getWidth() const;
    [[nodiscard]] std::pair<int,int> getStart() const;
    [[nodiscard]] std::pair<int,int> getEnd() const;
    [[nodiscard]] const std::vector<std::vector<bool>>& getMaze() const;
    [[nodiscard]] std::string getGeneratorName() const;
    void setGeneratorName(const std::string &name);

    void set(int h, int w, bool b);
    [[nodiscard]] bool get(int h, int w) const;
    [[nodiscard]] std::string mazeToStringSimple() const;
    [[nodiscard]] std::string mazeToString() const;
    void printSimple() const;
    void print() const;

    // Persistence: self-describing naming {seed}_{height}_{width}_{generator}.[png|maze]
    std::string save_maze(const std::string &custom_filepath = "");
    std::string save_binary(const std::string &custom_filepath = "") const;
    bool load_binary(const std::string &filepath);
};

#endif //MAZES_MAZE_HPP
