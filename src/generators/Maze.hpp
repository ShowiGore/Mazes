#ifndef MAZES_MAZE_HPP
#define MAZES_MAZE_HPP

#include <vector>
#include <array>
#include <stack>
#include <random>
#include <utility>
#include <iostream>
#include <png.hpp>

constexpr bool WALL = true;
constexpr bool PATH = false;
enum Direction {UP=0, RIGHT=1, DOWN=2, LEFT=3};

constexpr int N_DIRECTIONS = 4;

class Maze {

protected:

    std::vector<std::vector<bool>> maze;
    std::vector<std::vector<bool>> visited;
    std::vector<std::vector<bool>> solution;
    std::pair <int, int> start, end;
    int height, width;
    unsigned int seed;

    std::random_device rd;
    std::mt19937 re{};
    std::bernoulli_distribution bd;

    static int randomInRange (int min, int max);
    static int randomEven (int min, int max);
    static int randomOdd (int min, int max);
    virtual void init();
    void buildStartEnd();
    virtual void generate() {};

public:

    virtual ~Maze() = default;
    Maze(int height, int width, unsigned int seed);
    Maze(int height, int width);
    void set(int h, int w, bool b);
    bool get(int h, int w);
    std::string mazeToStringSimple();
    std::string mazeToString();
    void printSimple();
    void print();
    void save_maze();
    void save_solution();
    void solve();

};

#endif //MAZES_MAZE_HPP
