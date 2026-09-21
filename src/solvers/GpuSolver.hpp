#ifndef MAZES_GPUSOLVER_HPP
#define MAZES_GPUSOLVER_HPP

#include "Solver.hpp"
#include "utilities/GpuUtils.hpp"
#include <string>

/**
 * @brief Base class for all GPU-accelerated maze solvers
 *
 * Encapsulates common GPU device configuration, batch size control,
 * and adaptive execution rationales.
 */
class GpuSolver : public Solver {
protected:
    std::string preferred_device_vendor;
    std::string selected_platform_name;
    std::string selected_device_name;
    int user_batch_size = -1;
    std::string config_rationale;

public:
    explicit GpuSolver(std::string device_vendor = "any")
        : preferred_device_vendor(std::move(device_vendor)) {}
    ~GpuSolver() override = default;

    [[nodiscard]] std::string getDeviceName() const { return selected_device_name; }
    [[nodiscard]] std::string getPlatformName() const { return selected_platform_name; }
    [[nodiscard]] std::string getConfigRationale() const { return config_rationale; }

    void setBatchSize(int batch) { user_batch_size = batch; }
    [[nodiscard]] int getBatchSize() const { return user_batch_size; }
};

#endif // MAZES_GPUSOLVER_HPP
