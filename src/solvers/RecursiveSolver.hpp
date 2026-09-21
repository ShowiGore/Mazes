#ifndef MAZES_RECURSIVESOLVER_HPP
#define MAZES_RECURSIVESOLVER_HPP

#pragma once
#include "Solver.hpp"

class RecursiveSolver : public Solver {
public:
    RecursiveSolver() { this->solver_name = "recursive"; }
    bool solve(const Maze &maze) override;
};


#endif //MAZES_RECURSIVESOLVER_HPP