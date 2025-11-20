#include "RecursiveDivisionMaze.hpp"


RecursiveDivisionMaze::RecursiveDivisionMaze(int height, int width, unsigned int seed) : Maze(height, width, seed) {
    init();
    generate();
    buildStartEnd();
}

RecursiveDivisionMaze::RecursiveDivisionMaze(int height, int width) : Maze(height, width) {
    init();
    generate();
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

void RecursiveDivisionMaze::buildVertical (int w, int minH, int maxH) {
    if (w%2 != 0) {
        std::cout << "Error at biuldVertical()" << std::endl;
    } else {
        for (int i=minH; i<=maxH; i++) {
            maze[i][w] = WALL;
        }
    }

}

void RecursiveDivisionMaze::buildHorizontal (int h, int minW, int maxW) {
    if (h%2 != 0) {
        std::cout << "Error at biuldHorizontal()" << std::endl;
    } else {
        for (int j=minW; j<=maxW; j++) {
            maze[h][j] = WALL;
        }
    }
}

void RecursiveDivisionMaze::recursiveDivisionVertical (int minHeight, int maxHeight, int minWidth, int maxWidth) {
    int h = (maxHeight-minHeight)+1;
    int w = (maxWidth-minWidth)+1;

    if (h>3 && w>3) {

        int row = randomOdd(minHeight+1, maxHeight-1);
        int column = randomEven(minWidth+1, maxWidth-1);

        buildVertical(column, minHeight+1, maxHeight-1);
        maze[row][column] = PATH;

        recursiveDivisionRecursive(minHeight, maxHeight, minWidth, column);
        recursiveDivisionRecursive(minHeight, maxHeight, column, maxWidth);
    }
}

void RecursiveDivisionMaze::recursiveDivisionHorizontal (int minHeight, int maxHeight, int minWidth, int maxWidth) {
    int h = (maxHeight - minHeight) + 1;
    int w = (maxWidth - minWidth) + 1;

    if (h > 3 && w > 3) {
        int row = randomEven(minHeight + 1, maxHeight - 1);
        int column = randomOdd(minWidth + 1, maxWidth - 1);

        buildHorizontal(row, minWidth + 1, maxWidth - 1);
        maze[row][column] = PATH;

        recursiveDivisionRecursive(minHeight, row, minWidth, maxWidth);
        recursiveDivisionRecursive(row, maxHeight, minWidth, maxWidth);

    }
}

void RecursiveDivisionMaze::recursiveDivisionRecursive (int minHeight, int maxHeight, int minWidth, int maxWidth) {

    int h = (maxHeight - minHeight) + 1;
    int w = (maxWidth - minWidth) + 1;

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

