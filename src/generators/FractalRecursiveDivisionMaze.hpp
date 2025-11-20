#ifndef MAZES_FRACTALRECURSIVEDIVISIONMAZE_HPP
#define MAZES_FRACTALRECURSIVEDIVISIONMAZE_HPP

#include "Maze.hpp"

class FractalRecursiveDivisionMaze : public Maze {

protected:

    void init();
    void generate() override;

    void buildVertical (int w, int minH, int maxH);
    void buildHorizontal (int h, int minW, int maxW);
    void fractalRecursiveDivisionVertical (int minHeight, int maxHeight, int minWidth, int maxWidth);
    void fractalRecursiveDivisionHorizontal (int minHeight, int maxHeight, int minWidth, int maxWidth);
    void fractalRecursiveDivisionRecursive (int minHeight, int maxHeight, int minWidth, int maxWidth);

public:

    FractalRecursiveDivisionMaze(int height, int width, unsigned int seed);
    FractalRecursiveDivisionMaze(int height, int width);

};


#endif //MAZES_FRACTALRECURSIVEDIVISIONMAZE_HPP
