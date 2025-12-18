#include "RecursiveDivisionMaze.hpp"


RecursiveDivisionMaze::RecursiveDivisionMaze(const int height, const int width, const unsigned int seed) : Maze(height, width, seed) {
    RecursiveDivisionMaze::init();
    RecursiveDivisionMaze::generate();
    buildStartEnd();
}

RecursiveDivisionMaze::RecursiveDivisionMaze(const int height, const int width) : Maze(height, width) {
    RecursiveDivisionMaze::init();
    RecursiveDivisionMaze::generate();
    buildStartEnd();
}

void RecursiveDivisionMaze::init () {

    for (int h=0; h<height; h++) {
        maze[h][0] = WALL;
        maze[h][width-1] = WALL;
    }

    for (int w=0; w<width; w++) {
        maze[0][w] = WALL;
        maze[height-1][w] = WALL;
    }

}

void RecursiveDivisionMaze::buildVertical (const int w, const int minH, const int maxH) {
    if (w%2 != 0) {
        std::cout << "Error at biuldVertical()" << std::endl;
    } else {
        for (int i=minH; i<=maxH; i++) {
            maze[i][w] = WALL;
        }
    }

}

void RecursiveDivisionMaze::buildHorizontal (const int h, const int minW, const int maxW) {
    if (h%2 != 0) {
        std::cout << "Error at biuldHorizontal()" << std::endl;
    } else {
        for (int j=minW; j<=maxW; j++) {
            maze[h][j] = WALL;
        }
    }
}

void RecursiveDivisionMaze::recursiveDivisionVertical (const int minHeight, const int maxHeight, const int minWidth, const int maxWidth) {

    if (((maxHeight-minHeight)+1)>3 && ((maxWidth-minWidth)+1)>3) {

        const int row = randomOdd(minHeight+1, maxHeight-1);
        const int column = randomEven(minWidth+1, maxWidth-1);

        buildVertical(column, minHeight+1, maxHeight-1);
        maze[row][column] = PATH;

        recursiveDivisionRecursive(minHeight, maxHeight, minWidth, column);
        recursiveDivisionRecursive(minHeight, maxHeight, column, maxWidth);
    }
}

void RecursiveDivisionMaze::recursiveDivisionHorizontal (const int minHeight, const int maxHeight, const int minWidth, const int maxWidth) {
    const int h = (maxHeight - minHeight) + 1;
    const int w = (maxWidth - minWidth) + 1;

    if (h > 3 && w > 3) {
        const int row = randomEven(minHeight + 1, maxHeight - 1);
        const int column = randomOdd(minWidth + 1, maxWidth - 1);

        buildHorizontal(row, minWidth + 1, maxWidth - 1);
        maze[row][column] = PATH;

        recursiveDivisionRecursive(minHeight, row, minWidth, maxWidth);
        recursiveDivisionRecursive(row, maxHeight, minWidth, maxWidth);

    }
}

void RecursiveDivisionMaze::recursiveDivisionRecursive (const int minHeight, const int maxHeight, const int minWidth, const int maxWidth) {

    const int h = (maxHeight - minHeight) + 1;
    const int w = (maxWidth - minWidth) + 1;

    if (w < h) {
        recursiveDivisionHorizontal(minHeight, maxHeight, minWidth, maxWidth);
    } else if (h < w) {
        recursiveDivisionVertical(minHeight, maxHeight, minWidth, maxWidth);
    } else {
        if (rand()%2) {
            recursiveDivisionVertical(minHeight, maxHeight, minWidth, maxWidth);
        } else {
            recursiveDivisionHorizontal(minHeight, maxHeight, minWidth, maxWidth);
        }
    }
}

void RecursiveDivisionMaze::generate () {

    recursiveDivisionRecursive(0, height-1, 0, width-1);

}

