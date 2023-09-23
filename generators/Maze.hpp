#ifndef MAZES_MAZE_HPP
#define MAZES_MAZE_HPP

#include <vector>
#include <random>
#include <utility>
#include <iostream>
#include <png++/png.hpp>

const bool WALL = true;
const bool PATH = false;

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

    int randomInRange (int min, int max);
    int randomEven (int min, int max);
    int randomOdd (int min, int max);
    void init();
    void buildStartEnd();
    virtual void generate() {};

public:

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

};

#endif //MAZES_MAZE_HPP
