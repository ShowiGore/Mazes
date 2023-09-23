
#ifndef MAZES_TIMEPROFILER_HPP
#define MAZES_TIMEPROFILER_HPP

#include <chrono>
#include <string>
#include <iostream>

class TimeProfiler {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mEnd;
public:
    void start();
    void stop();
    void print();
};

#endif //MAZES_TIMEPROFILER_HPP
