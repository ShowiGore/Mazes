    #ifndef MAZES_WILSONSMAZE_HPP
#define MAZES_WILSONSMAZE_HPP

#include "Maze.hpp"

class WilsonsMaze : public Maze {

    private:
        enum WalkDirections { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };
        std::vector<std::vector<WalkDirections>> walk_grid;

        void init();
        void generate() override;
        void loopErasedRandomWalk(int start_row, int start_column);

    public:
        WilsonsMaze(int height, int width, unsigned int seed);
        WilsonsMaze(int height, int width);

};


#endif //MAZES_WILSONSMAZE_HPP
