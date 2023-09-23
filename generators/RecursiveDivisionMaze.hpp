#ifndef MAZES_RECURSIVEDIVISIONMAZE_HPP
#define MAZES_RECURSIVEDIVISIONMAZE_HPP

#include "Maze.hpp"

class RecursiveDivisionMaze : public Maze {

protected:

    void init();
    void generate() override;

    void buildVertical (int w, int minH, int maxH);
    void buildHorizontal (int h, int minW, int maxW);
    void recursiveDivisionVertical (int minHeight, int maxHeight, int minWidth, int maxWidth);
    void recursiveDivisionHorizontal (int minHeight, int maxHeight, int minWidth, int maxWidth);
    void recursiveDivisionRecursive (int minHeight, int maxHeight, int minWidth, int maxWidth);

public:

    RecursiveDivisionMaze(int height, int width, unsigned int seed);
    RecursiveDivisionMaze(int height, int width);

};


#endif //MAZES_RECURSIVEDIVISIONMAZE_HPP
