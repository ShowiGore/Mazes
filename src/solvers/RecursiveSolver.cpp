#include "RecursiveSolver.hpp"


bool RecursiveSolver::solve(const Maze &maze_object) { //dfs

    this->height = maze_object.getHeight();
    this->width = maze_object.getWidth();

    this->start = maze_object.getStart();
    this->end = maze_object.getEnd();

    const std::vector<std::vector<bool>> &maze = maze_object.getMaze();

    this->visited.assign(this->height, std::vector<bool>(this->width, false));
    this->solution.assign(this->height, std::vector<bool>(this->width, false));

    constexpr Direction directions[] = {UP, RIGHT, DOWN, LEFT};
    std::stack<Direction> steps;

    std::pair <int, int> current = start;
    visited[current.first][current.second] = true;

    int lastStep = -1;


    //Search end
    while (current != end && !(lastStep == N_DIRECTIONS-1 && current == start)) {

        bool stepped = false;
        int d = lastStep+1;
        while (d <= N_DIRECTIONS && !stepped) {
            if (d == N_DIRECTIONS) { //backtrack
                lastStep = steps.top();
                steps.pop();
                switch (lastStep) {
                    case UP: current.first = current.first+1; break;
                    case RIGHT: current.second = current.second-1; break;
                    case DOWN: current.first = current.first-1; break;
                    case LEFT: current.second = current.second+1; break;
                    default: break;
                }
            } else {
                switch (directions[d]) {
                    case UP:
                        if (current.first > 0 && maze[current.first-1][current.second] == PATH && !visited[current.first-1][current.second]) {
                            steps.push(UP);
                            current.first = current.first-1;
                            visited[current.first][current.second] = true;
                            lastStep = -1;
                            stepped = true;
                            //std::cout << "go UP" << std::endl;
                        }
                        break;
                    case RIGHT:
                        if (current.second < width-1 && maze[current.first][current.second+1] == PATH && !visited[current.first][current.second+1]) {
                            steps.push(RIGHT);
                            current.second = current.second+1;
                            visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go RIGHT" << std::endl;
                        }
                        break;
                    case DOWN:
                        if (current.first < height-1 && maze[current.first+1][current.second] == PATH && !visited[current.first+1][current.second]) {
                            steps.push(DOWN);
                            current.first = current.first+1;
                            visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go DOWN" << std::endl;
                        }
                        break;
                    case LEFT:
                        if (current.second > 0 && maze[current.first][current.second-1] == PATH && !visited[current.first][current.second-1]) {
                            steps.push(LEFT);
                            current.second = current.second-1;
                            visited[current.first][current.second] = true;
                            stepped = true;
                            lastStep = -1;
                            //std::cout << "go LEFT" << std::endl;
                        }
                        break;
                }
            }
            ++d;
        }

    }

    //Build solution
    if (current == end) {
        solution[current.first][current.second] = true;
        while (!steps.empty()) {
            lastStep = steps.top();
            steps.pop();
            switch (lastStep) {
                case UP: current.first = current.first+1; break;
                case RIGHT: current.second = current.second-1; break;
                case DOWN: current.first = current.first-1; break;
                case LEFT: current.second = current.second+1; break;
                default: break;
            }
            solution[current.first][current.second] = true;
        }
        return true;
    }

    return false;

}
