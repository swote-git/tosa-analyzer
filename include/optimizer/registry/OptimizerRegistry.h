#ifndef OPTIMIZER_REGISTRY_H
#define OPTIMIZER_REGISTRY_H

#include "optimizer/MemoryOptimizerInterface.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace optimizer {
namespace registry {

class OptimizerRegistry {
public:
    using CreatorFunction = std::function<std::unique_ptr<MemoryOptimizerInterface>()>;
    
    static OptimizerRegistry& getInstance();
    
    // Register an optimizer
    void registerOptimizer(const std::string& name, CreatorFunction creator);
    
    // Create an optimizer by name
    std::unique_ptr<MemoryOptimizerInterface> createOptimizer(const std::string& name);
    
    // Get list of available optimizers
    std::vector<std::string> getAvailableOptimizers() const;

private:
    OptimizerRegistry() = default;
    std::map<std::string, CreatorFunction> registry;
};

} // namespace registry
} // namespace optimizer

// Helper function for optimizer registration
// This approach avoids using macros which can be error-prone
namespace optimizer {
namespace registry {

template <typename OptimizerClass>
bool registerOptimizer(const std::string& name) {
    OptimizerRegistry::getInstance().registerOptimizer(
        name, 
        []() { return std::make_unique<OptimizerClass>(); }
    );
    return true;
}

} // namespace registry
} // namespace optimizer

#endif // OPTIMIZER_REGISTRY_H