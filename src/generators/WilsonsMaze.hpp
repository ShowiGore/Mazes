    #ifndef MAZES_WILSONSMAZE_HPP
#define MAZES_WILSONSMAZE_HPP

#include "Maze.hpp"

class WilsonsMaze : public Maze {

    private:
        std::vector<std::vector<Direction>> walk_grid;

        void init() override;
        void generate() override;
        void loop_erased_random_walk_initial(int start_row, int start_column);
        void loopErasedRandomWalk(int start_row, int start_column);

    public:
        WilsonsMaze(int height, int width, unsigned int seed);
        WilsonsMaze(int height, int width);

};


#endif //MAZES_WILSONSMAZE_HPP
