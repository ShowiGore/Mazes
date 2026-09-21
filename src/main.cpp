#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <random>
#include <memory>
#include <iomanip>

#include "utilities/TimeProfiler.hpp"

// Generators
#include "generators/Maze.hpp"
#include "generators/HoustonMaze.hpp"
#include "generators/WilsonsMaze.hpp"
#include "generators/AldousBroderMaze.hpp"
#include "generators/RecursiveDivisionMaze.hpp"
#include "generators/FractalRecursiveDivisionMaze.hpp"

// Solvers
#include "solvers/Solver.hpp"
#include "solvers/BidirectionalGreedyBestFirstSolver.hpp"
#include "solvers/GreedyBestFirstSolver.hpp"
#include "solvers/DeadEndFillingSolver.hpp"
#include "solvers/AStarSolver.hpp"
#include "solvers/RecursiveSolver.hpp"

struct Config {
    int height = 8193;
    int width = 8193;
    bool height_specified = false;
    bool width_specified = false;

    unsigned int seed = 0;
    bool seed_specified = false;

    std::string generator = "";
    bool generator_specified = false;

    float houston_alpha = 1.0f / 3.0f;
    bool alpha_specified = false;

    std::vector<std::string> solvers;
    bool solvers_specified = false;

    bool save_png = false;
    bool save_bin = false;
    std::string load_filepath = "";
};

void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n"
              << "General Options:\n"
              << "  -h, --help                 Show this help message and exit\n"
              << "  -H, --height <int>         Maze height (default: 8193, must be odd >= 3)\n"
              << "  -W, --width <int>          Maze width (default: 8193, must be odd >= 3)\n"
              << "  -s, --seed <uint>          Random seed (default: random)\n"
              << "  --png                      Save PNG image(s) to generated_mazes/\n"
              << "  --bin                      Save compact binary .maze / .sol file(s)\n\n"
              << "Generator Options (Required unless --load is specified):\n"
              << "  -g, --gen <name>           Generator to use:\n"
              << "                               houston             (Aldous-Broder + Wilson hybrid)\n"
              << "                               wilson              (Wilson's LERW algorithm)\n"
              << "                               aldous-broder       (Aldous-Broder algorithm)\n"
              << "                               recursive-division  (Recursive Division)\n"
              << "                               fractal             (Fractal Recursive Division)\n"
              << "  -a, --alpha <float>        Transition threshold for Houston (default: 0.333)\n\n"
              << "Solver Options (Optional - solving is skipped if omitted):\n"
              << "  --solvers, --solver <list> Comma-separated list of solvers, or 'all', or 'none':\n"
              << "                               bidir-gbfs          (Bidirectional Greedy BFS - minimum visited cells)\n"
              << "                               gbfs                (Greedy Best-First Search)\n"
              << "                               dead-end            (Deterministic O(N) Dead-End Filling)\n"
              << "                               astar               (A* search)\n"
              << "                               recursive           (Recursive DFS)\n"
              << "                               all                 (Run all solvers and compare)\n"
              << "                               none                (Skip solving)\n\n"
              << "Decoupled Mode:\n"
              << "  --load <file>              Load existing binary .maze file and solve directly\n\n"
              << "Examples:\n"
              << "  " << prog_name << " -g houston\n"
              << "  " << prog_name << " -g houston -H 2001 -W 2001 -s 42 --solvers bidir-gbfs --png\n"
              << "  " << prog_name << " -g wilson -H 1001 -W 1001 --solvers all\n"
              << "  " << prog_name << " --load generated_mazes/42_2001_2001_houston.maze --solvers dead-end\n";
}

int main(int argc, char* argv[]) {
    Config config;

    // Default random seed
    std::random_device rd;
    config.seed = rd();

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if ((arg == "-H" || arg == "--height") && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
            config.height_specified = true;
        } else if ((arg == "-W" || arg == "--width") && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
            config.width_specified = true;
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            config.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
            config.seed_specified = true;
        } else if ((arg == "-g" || arg == "--gen") && i + 1 < argc) {
            config.generator = argv[++i];
            std::transform(config.generator.begin(), config.generator.end(), config.generator.begin(), ::tolower);
            config.generator_specified = true;
        } else if ((arg == "-a" || arg == "--alpha") && i + 1 < argc) {
            config.houston_alpha = std::stof(argv[++i]);
            config.alpha_specified = true;
        } else if ((arg == "--solvers" || arg == "--solver") && i + 1 < argc) {
            std::string solver_arg = argv[++i];
            config.solvers_specified = true;
            std::stringstream ss(solver_arg);
            std::string item;
            while (std::getline(ss, item, ',')) {
                std::transform(item.begin(), item.end(), item.begin(), ::tolower);
                config.solvers.push_back(item);
            }
        } else if (arg == "--png") {
            config.save_png = true;
        } else if (arg == "--bin") {
            config.save_bin = true;
        } else if (arg == "--load" && i + 1 < argc) {
            config.load_filepath = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\nUse --help for usage.\n";
            return 1;
        }
    }

    // Validation: generator must be specified unless --load is provided
    if (config.load_filepath.empty() && !config.generator_specified) {
        std::cerr << "Error: No generator specified!\n"
                  << "You must specify a generator using -g or --gen (e.g. -g houston), or load an existing maze using --load <file>.\n\n";
        print_help(argv[0]);
        return 1;
    }

    // Ensure dimensions are odd and >= 3
    if (config.height < 3) config.height = 3;
    if (config.width < 3) config.width = 3;
    if (!(config.height & 1)) config.height += 1;
    if (!(config.width & 1)) config.width += 1;

    // =========================================================================
    // EXECUTION CONFIGURATION SUMMARY
    // =========================================================================
    std::cout << "============================================================\n"
              << "MAZE ENGINE EXECUTION CONFIGURATION\n"
              << "============================================================\n";

    if (!config.load_filepath.empty()) {
        std::cout << "Mode:               Load Binary Maze & Solve\n"
                  << "Source File:        " << config.load_filepath << "\n";
    } else {
        std::cout << "Mode:               Generate Maze"
                  << (config.solvers.empty() ? " (solving skipped)\n" : " & Solve\n")
                  << "Generator:          " << config.generator << " [specified]\n"
                  << "Dimensions:         " << config.height << " x " << config.width
                  << (config.height_specified || config.width_specified ? " [specified]\n" : " [default: 8193]\n")
                  << "Total Grid Cells:   " << static_cast<size_t>(config.height) * config.width << "\n"
                  << "Graph Nodes:        " << (static_cast<size_t>(config.height) / 2) * (config.width / 2) << "\n"
                  << "Seed:               " << config.seed
                  << (config.seed_specified ? " [specified]\n" : " [random]\n");

        if (config.generator == "houston") {
            std::cout << "Houston Alpha:      " << config.houston_alpha
                      << (config.alpha_specified ? " [specified]\n" : " [default: 0.333]\n");
        }
    }

    std::cout << "Solvers:            ";
    if (config.solvers.empty() || (config.solvers.size() == 1 && config.solvers[0] == "none")) {
        std::cout << "[none - solving will be skipped]\n";
    } else {
        for (size_t i = 0; i < config.solvers.size(); ++i) {
            std::cout << config.solvers[i] << (i + 1 < config.solvers.size() ? ", " : " [specified]\n");
        }
    }

    std::cout << "Save PNG:           " << (config.save_png ? "YES" : "NO [default]") << "\n"
              << "Save Binary:        " << (config.save_bin ? "YES" : "NO [default]") << "\n"
              << "Output Directory:   generated_mazes/\n"
              << "============================================================\n\n";

    TimeProfiler tp;
    std::shared_ptr<Maze> maze;

    // =========================================================================
    // STEP 1: OBTAIN MAZE (LOAD OR GENERATE)
    // =========================================================================
    if (!config.load_filepath.empty()) {
        std::cout << "Loading maze from: " << config.load_filepath << " ..." << std::endl;
        maze = std::make_shared<Maze>();
        tp.start();
        if (!maze->load_binary(config.load_filepath)) {
            std::cerr << "Failed to load maze from " << config.load_filepath << std::endl;
            return 1;
        }
        tp.stop();
        std::cout << "Maze loaded successfully in: ";
        tp.print();
        config.height = maze->getHeight();
        config.width = maze->getWidth();
    } else {
        std::cout << "Generating maze using '" << config.generator << "'..." << std::endl;
        tp.start();

        if (config.generator == "houston") {
            maze = std::make_shared<HoustonMaze>(config.height, config.width, config.seed, config.houston_alpha);
        } else if (config.generator == "wilson") {
            maze = std::make_shared<WilsonsMaze>(config.height, config.width, config.seed);
        } else if (config.generator == "aldous-broder") {
            maze = std::make_shared<AldousBroderMaze>(config.height, config.width, config.seed);
        } else if (config.generator == "recursive-division") {
            auto r_maze = std::make_shared<RecursiveDivisionMaze>(config.height, config.width, config.seed);
            r_maze->setGeneratorName("recursive-division");
            maze = r_maze;
        } else if (config.generator == "fractal") {
            auto f_maze = std::make_shared<FractalRecursiveDivisionMaze>(config.height, config.width, config.seed);
            f_maze->setGeneratorName("fractal");
            maze = f_maze;
        } else {
            std::cerr << "Error: unknown generator '" << config.generator << "'.\nUse --help for available generators.\n";
            return 1;
        }

        tp.stop();
        std::cout << "Maze generation time: ";
        tp.print();

        if (config.save_bin) {
            std::string bin_path = maze->save_binary();
            std::cout << "Saved binary maze to: " << bin_path << std::endl;
        }
        if (config.save_png) {
            std::cout << "Rendering maze PNG..." << std::endl;
            std::string png_path = maze->save_maze();
            std::cout << "Saved maze image to: " << png_path << std::endl;
        }
    }

    // =========================================================================
    // STEP 2: SOLVE MAZE (OPTIONAL)
    // =========================================================================
    if (config.solvers.empty() || (config.solvers.size() == 1 && config.solvers[0] == "none")) {
        std::cout << "\nSolving skipped (no solvers requested).\n";
        return 0;
    }

    // Expand 'all' into full list of solvers
    std::vector<std::string> active_solvers;
    for (const auto& s : config.solvers) {
        if (s == "all") {
            active_solvers = {"bidir-gbfs", "gbfs", "dead-end", "astar", "recursive"};
            break;
        }
        active_solvers.push_back(s);
    }

    std::cout << "\n============================================================\n"
              << "SOLVING MAZE\n"
              << "============================================================\n";

    for (const auto& solver_name : active_solvers) {
        std::unique_ptr<Solver> solver;

        if (solver_name == "bidir-gbfs" || solver_name == "bidirectional-gbfs") {
            solver = std::make_unique<BidirectionalGreedyBestFirstSolver>();
        } else if (solver_name == "gbfs" || solver_name == "greedy-bfs") {
            solver = std::make_unique<GreedyBestFirstSolver>();
        } else if (solver_name == "dead-end") {
            solver = std::make_unique<DeadEndFillingSolver>();
        } else if (solver_name == "astar") {
            solver = std::make_unique<AStarSolver>();
        } else if (solver_name == "recursive") {
            solver = std::make_unique<RecursiveSolver>();
        } else {
            std::cerr << "Warning: unknown solver '" << solver_name << "', skipping.\n";
            continue;
        }

        std::cout << "\n--- Solver: " << solver->getSolverName() << " ---" << std::endl;
        tp.start();
        const bool solvable = solver->solve(*maze);
        tp.stop();

        std::cout << "Solve time: ";
        tp.print();
        std::cout << "Solvable:   " << (solvable ? "YES" : "NO") << std::endl;

        if (auto* bidir = dynamic_cast<BidirectionalGreedyBestFirstSolver*>(solver.get())) {
            std::cout << "Cells visited: " << bidir->getVisitedCount() << std::endl;
        }

        if (solvable) {
            if (config.save_bin) {
                std::string sol_bin = solver->save_binary_solution(*maze);
                std::cout << "Saved binary solution: " << sol_bin << std::endl;
            }
            if (config.save_png) {
                std::cout << "Rendering solution PNG..." << std::endl;
                std::string sol_png = solver->save_solution(*maze);
                std::cout << "Saved solution image:  " << sol_png << std::endl;
            }
        }
    }

    std::cout << "\nAll operations completed successfully!\n";
    return 0;
}
