    #ifndef MAZES_WILSONNMAZE_HPP
#define MAZES_WILSONNMAZE_HPP

#include "Maze.hpp"

class WilsonMaze : public Maze {

protected:

    void init();
    void generate() override;

    void buildVertical (int w, int minH, int maxH);
    void buildHorizontal (int h, int minW, int maxW);
    void recursiveDivisionVertical (int minHeight, int maxHeight, int minWidth, int maxWidth);
    void recursiveDivisionHorizontal (int minHeight, int maxHeight, int minWidth, int maxWidth);
    void recursiveDivisionRecursive (int minHeight, int maxHeight, int minWidth, int maxWidth);

public:

    WilsonMaze(int height, int width, unsigned int seed);
    WilsonMaze(int height, int width);

};


#endif //MAZES_WILSONNMAZE_HPP
