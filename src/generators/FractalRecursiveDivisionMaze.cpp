#include "FractalRecursiveDivisionMaze.hpp"


FractalRecursiveDivisionMaze::FractalRecursiveDivisionMaze(int height, int width, unsigned int seed) : Maze(height, width, seed) {
    init();
    generate();
    buildStartEnd();
}

FractalRecursiveDivisionMaze::FractalRecursiveDivisionMaze(int height, int width) : Maze(height, width) {
    init();
    generate();
    buildStartEnd();
}

void FractalRecursiveDivisionMaze::init () {

    for (int h=0; h<height; h++) {
        maze[h][0] = WALL;
        maze[h][width-1] = WALL;
    }

    for (int w=0; w<width; w++) {
        maze[0][w] = WALL;
        maze[height-1][w] = WALL;
    }

}

void FractalRecursiveDivisionMaze::buildVertical (int w, int minH, int maxH) {
    if (w%2 != 0) {
        std::cout << "Error at biuldVertical()" << std::endl;
    } else {
        for (int i=minH; i<=maxH; i++) {
            maze[i][w] = WALL;
        }
    }

}

void FractalRecursiveDivisionMaze::buildHorizontal (int h, int minW, int maxW) {
    if (h%2 != 0) {
        std::cout << "Error at biuldHorizontal()" << std::endl;
    } else {
        for (int j=minW; j<=maxW; j++) {
            maze[h][j] = WALL;
        }
    }
}

void FractalRecursiveDivisionMaze::fractalRecursiveDivisionVertical (int minHeight, int maxHeight, int minWidth, int maxWidth) {
    int h = (maxHeight-minHeight)+1;
    int w = (maxWidth-minWidth)+1;

    if (h>3 && w>3) {
        int row, column;

        row = (maxHeight-minHeight)/2 + minHeight+1;
        column = (maxWidth-minWidth)/2 + minWidth+1;

        if(row%2==0) {
            row += 1;
        }
        if(column%2==1) {
            column -= 1;
        }

        buildVertical(column, minHeight+1, maxHeight-1);
        maze[row][column] = PATH;

        fractalRecursiveDivisionHorizontal(minHeight, maxHeight, minWidth, column);
        fractalRecursiveDivisionHorizontal(minHeight, maxHeight, column, maxWidth);
    }
}

void FractalRecursiveDivisionMaze::fractalRecursiveDivisionHorizontal (int minHeight, int maxHeight, int minWidth, int maxWidth) {
    int h = (maxHeight-minHeight)+1;
    int w = (maxWidth-minWidth)+1;

    if (h>3 && w>3) {
        int row, column;

        row = (maxHeight-minHeight)/2 + minHeight+1;
        column = (maxWidth-minWidth)/2 + minWidth+1;

        if(row%2==1) {
            row -= 1;
        }
        if(column%2==0) {
            column += 1;
        }

        buildHorizontal(row, minWidth+1, maxWidth-1);
        maze[row][column] = PATH;

        fractalRecursiveDivisionVertical(minHeight, row, minWidth, maxWidth);
        fractalRecursiveDivisionVertical(row, maxHeight, minWidth, maxWidth);

    }
}

void FractalRecursiveDivisionMaze::fractalRecursiveDivisionRecursive (int minHeight, int maxHeight, int minWidth, int maxWidth) {

    int h = (maxHeight - minHeight) + 1;
    int w = (maxWidth - minWidth) + 1;

    if (w < h) {
        fractalRecursiveDivisionHorizontal(minHeight, maxHeight, minWidth, maxWidth);
    } else if (h < w) {
        fractalRecursiveDivisionVertical(minHeight, maxHeight, minWidth, maxWidth);
    } else {
        if (rand()%2) {
            fractalRecursiveDivisionVertical(minHeight, maxHeight, minWidth, maxWidth);
        } else {
            fractalRecursiveDivisionHorizontal(minHeight, maxHeight, minWidth, maxWidth);
        }
    }

}

void FractalRecursiveDivisionMaze::generate () {

    fractalRecursiveDivisionRecursive(0, height-1, 0, width-1);

}

