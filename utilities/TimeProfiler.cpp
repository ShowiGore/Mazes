#include "TimeProfiler.hpp"

void TimeProfiler::start(){
    mStart = std::chrono::high_resolution_clock::now();
}

void TimeProfiler::stop(){
    mEnd = std::chrono::high_resolution_clock::now();
}

void TimeProfiler::print() {

    long t = std::chrono::duration_cast<std::chrono::nanoseconds>(mEnd - mStart).count();

    auto nanoseconds = t%1000;
    t = t/1000;
    long microseconds = t%1000;
    t = t/1000;
    long milliseconds = t%1000;
    t = t/1000;
    long seconds = t%60;
    t = t/60;
    long minutes = t%60;
    t = t/60;
    long hours = t%24;
    t = t/24;
    long days = t;

    std::string s = "[";
    if (days>0) {s.append(std::to_string(days) + " d | ");}
    if (hours>0) {s.append(std::to_string(hours) + " h | ");}
    if (minutes>0) {s.append(std::to_string(minutes) + " min | ");}
    if (seconds>0) {s.append(std::to_string(seconds) + " s | ");}
    if (milliseconds>0) {s.append(std::to_string(milliseconds) + " ms | ");}
    if (microseconds>0) {s.append(std::to_string(microseconds) + " µs | ");}
    if (nanoseconds>0) {s.append(std::to_string(nanoseconds) + " ns | ");}

    s.erase(s.length()-3);
    s.append("]");

    std::cout << s << std::endl;

}
