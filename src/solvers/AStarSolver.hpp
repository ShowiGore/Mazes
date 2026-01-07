#ifndef MAZES_ASTARSOLVER_HPP
#define MAZES_ASTARSOLVER_HPP

#pragma once
#include "Solver.hpp"

class AStarSolver : public Solver {

public:

    bool solve(const Maze &maze) override;

};

#endif //MAZES_ASTARSOLVER_HPP