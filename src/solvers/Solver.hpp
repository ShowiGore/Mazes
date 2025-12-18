#ifndef MAZES_SOLVER_HPP
#define MAZES_SOLVER_HPP

#pragma once
#include <vector>
#include <utility>
#include "../generators/Maze.hpp"

class Solver {
    protected:
        std::vector<std::vector<bool>> visited;
        std::vector<std::vector<bool>> solution;
        std::pair <int, int> start, end;
        int height = 0, width = 0;
    public:
        virtual ~Solver() = default;
        virtual bool solve(const Maze &maze) = 0;
        void save_solution(const Maze &maze);
};

#endif //MAZES_SOLVER_HPP
