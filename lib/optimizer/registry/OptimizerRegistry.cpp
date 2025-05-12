#include "optimizer/registry/OptimizerRegistry.h"

namespace optimizer {
namespace registry {

OptimizerRegistry& OptimizerRegistry::getInstance() {
    static OptimizerRegistry instance;
    return instance;
}

void OptimizerRegistry::registerOptimizer(const std::string& name, CreatorFunction creator) {
    registry[name] = creator;
}

std::unique_ptr<MemoryOptimizerInterface> OptimizerRegistry::createOptimizer(const std::string& name) {
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> OptimizerRegistry::getAvailableOptimizers() const {
    std::vector<std::string> result;
    for (const auto& entry : registry) {
        result.push_back(entry.first);
    }
    return result;
}

} // namespace registry
} // namespace optimizer