#ifndef MAZES_HOUSTONMAZE_HPP
#define MAZES_HOUSTONMAZE_HPP

#include "Maze.hpp"

/**
 * @brief Houston's Maze Generator (Aldous-Broder / Wilson Hybrid)
 *
 * Combines the early-game speed of Aldous-Broder with the late-game speed
 * of Wilson's algorithm, preserving the Uniform Spanning Tree (UST) property.
 *
 * Constructors:
 * 1. Without alpha: HoustonMaze(height, width[, seed]) -> Automatically uses THEORETICAL_ALPHA (1/3).
 * 2. With alpha:    HoustonMaze(height, width, seed, alpha) or HoustonMaze(height, width, alpha).
 */
class HoustonMaze : public Maze {
public:
    /// Theoretical optimal transition threshold: alpha* ≈ 1/3 (Robin Houston, Jamis Buck)
    static constexpr float THEORETICAL_ALPHA = 1.0f / 3.0f;

private:
    float switch_threshold; // Fraction of cells (alpha) at which to switch to Wilson's algorithm

    void init() override;
    void generate() override;

public:
    // Constructors using the theoretical optimal threshold (THEORETICAL_ALPHA = 1/3)
    HoustonMaze(int height, int width, unsigned int seed);
    HoustonMaze(int height, int width);

    // Constructors with a custom transition threshold alpha in (0, 1)
    HoustonMaze(int height, int width, unsigned int seed, float switch_threshold);
    HoustonMaze(int height, int width, float switch_threshold);

    [[nodiscard]] float getSwitchThreshold() const { return switch_threshold; }
};

#endif // MAZES_HOUSTONMAZE_HPP
