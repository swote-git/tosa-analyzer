#ifndef MEMORY_ALLOCATION_PLAN_H
#define MEMORY_ALLOCATION_PLAN_H

#include "TensorLifetimeInfo.h"
#include <vector>
#include <map>
#include <string>

namespace optimizer {

// Represents a memory pool in the allocation plan
struct MemoryPool {
    std::string name;
    size_t size;
    
    // Constructor
    MemoryPool(const std::string& name, size_t size) 
        : name(name), size(size) {}
};

// Represents a single tensor allocation
struct TensorAllocation {
    // Reference to the tensor info
    const TensorLifetimeInfo* tensor;
    
    // Which pool this tensor is allocated in
    int poolIndex;
    
    // Offset within the memory pool
    size_t offset;
    
    // Constructor
    TensorAllocation(const TensorLifetimeInfo* tensor, int poolIndex, size_t offset)
        : tensor(tensor), poolIndex(poolIndex), offset(offset) {}
};

// The complete memory allocation plan
class MemoryAllocationPlan {
public:
    // Add a new memory pool
    int addMemoryPool(const std::string& name, size_t size) {
        pools.emplace_back(name, size);
        return pools.size() - 1;
    }
    
    // Allocate a tensor in a specific pool
    void allocateTensor(const TensorLifetimeInfo* tensor, int poolIndex, size_t offset) {
        allocations.emplace_back(tensor, poolIndex, offset);
        tensorToAllocation[tensor->id] = allocations.size() - 1;
    }
    
    // Get allocation for a specific tensor
    const TensorAllocation* getAllocationForTensor(const std::string& tensorId) const {
        auto it = tensorToAllocation.find(tensorId);
        if (it != tensorToAllocation.end()) {
            return &allocations[it->second];
        }
        return nullptr;
    }
    
    // Get all memory pools
    const std::vector<MemoryPool>& getMemoryPools() const {
        return pools;
    }
    
    // Get all allocations
    const std::vector<TensorAllocation>& getAllocations() const {
        return allocations;
    }
    
    // Calculate total memory usage (sum of all pool sizes)
    size_t getTotalMemoryUsage() const {
        size_t total = 0;
        for (const auto& pool : pools) {
            total += pool.size;
        }
        return total;
    }
    
    // Clear the plan
    void clear() {
        pools.clear();
        allocations.clear();
        tensorToAllocation.clear();
    }
    
private:
    // Memory pools in this plan
    std::vector<MemoryPool> pools;
    
    // All tensor allocations
    std::vector<TensorAllocation> allocations;
    
    // Map from tensor ID to its allocation index
    std::map<std::string, size_t> tensorToAllocation;
};

} // namespace optimizer

#endif // MEMORY_ALLOCATION_PLAN_H