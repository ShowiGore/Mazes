#ifndef MAZES_SOLVER_HPP
#define MAZES_SOLVER_HPP

#pragma once
#include <vector>
#include <utility>
#include <string>
#include "../generators/Maze.hpp"

class Solver {
protected:
    std::vector<std::vector<bool>> visited;
    std::vector<std::vector<bool>> solution;
    std::pair<int, int> start, end;
    int height = 0, width = 0;
    std::string solver_name = "unknown";

public:
    virtual ~Solver() = default;
    virtual bool solve(const Maze &maze) = 0;

    [[nodiscard]] std::string getSolverName() const { return solver_name; }
    void setSolverName(const std::string &name) { solver_name = name; }

    // Self-describing naming: {seed}_{height}_{width}_{generator}_{solver}.png
    std::string save_solution(const Maze &maze, const std::string &custom_filepath = "");

    // Compact binary solution persistence: {seed}_{height}_{width}_{generator}_{solver}.sol
    std::string save_binary_solution(const Maze &maze, const std::string &custom_filepath = "") const;
};

#endif //MAZES_SOLVER_HPP
