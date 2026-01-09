#ifndef MAZES_ASTARSOLVER_HPP
#define MAZES_ASTARSOLVER_HPP

#pragma once
#include "Solver.hpp"

class AStarSolver : public Solver {

private:
    int heuristic_weight = 1;

    template <bool IsWeighted>
    bool solve_impl(const Maze &maze_object);

public:

    bool solve(const Maze &maze) override;

    void setHeuristicWeight(const int weight) { heuristic_weight = weight; }

};

#endif //MAZES_ASTARSOLVER_HPP