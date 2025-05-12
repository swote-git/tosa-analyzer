#include "BaseMemoryOptimizer.cpp"
#include "optimizer/registry/OptimizerRegistry.h"

namespace optimizer {
namespace strategies {
    class GreedyFirstFit : public BaseMemoryOptimizer {
        public:
        std::string getName() const override {
        return "GreedyFirstFit";
    }
    
    std::string getDescription() const override {
        return "A simple greedy algorithm that allocates tensors using a first-fit "
        "strategy based on their definition order.";
    }
    
    MemoryAllocationPlan optimize(
        const std::vector<TensorLifetimeInfo>& tensors) override {
            
            MemoryAllocationPlan plan;
            
            // Measure execution time
            measureExecutionTime([&]() {
            // Create a IO pool for inputs/outputs
            int ioPoolIdx = createInputOutputPool(plan, tensors);
            
            // Create main memory pool
            int mainPoolIdx = plan.addMemoryPool("main_pool", 1024 * 1024); // Start with 1MB
            
            // Sort tensors by definition point
            std::vector<const TensorLifetimeInfo*> sortedTensors;
            for (const auto& tensor : tensors) {
                sortedTensors.push_back(&tensor);
            }
            
            std::sort(sortedTensors.begin(), sortedTensors.end(),
                [](const TensorLifetimeInfo* a, const TensorLifetimeInfo* b) {
                    return a->defPoint < b->defPoint;
                });
            
                // Track free memory regions in the main pool
                struct FreeRegion {
                    size_t offset;
                    size_t size;
                };
                std::vector<FreeRegion> freeRegions = {{0, 1024 * 1024}}; // Initially all free
                
                // Track allocated tensors and their lifetimes
                struct AllocatedTensor {
                    const TensorLifetimeInfo* tensor;
                    size_t offset;
                };
                std::vector<AllocatedTensor> allocatedTensors;
                
                // For each tensor in definition order
                for (const TensorLifetimeInfo* tensor : sortedTensors) {
                // Handle inputs/outputs separately
                if (tensor->isModelInput || tensor->isModelOutput) {
                    // For simplicity, inputs/outputs go at offset 0 in IO pool
                    plan.allocateTensor(tensor, ioPoolIdx, 0);
                    continue;
                }
                
                // First, free any tensors that are no longer needed
                for (auto it = allocatedTensors.begin(); it != allocatedTensors.end();) {
                    if (it->tensor->lastUsePoint < tensor->defPoint) {
                        // This tensor is no longer needed, free its memory
                        FreeRegion newRegion = {it->offset, it->tensor->size};
                        freeRegions.push_back(newRegion);
                        
                        // Remove from allocated list
                        it = allocatedTensors.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                // Merge adjacent free regions
                if (!freeRegions.empty()) {
                    std::sort(freeRegions.begin(), freeRegions.end(),
                        [](const FreeRegion& a, const FreeRegion& b) {
                            return a.offset < b.offset;
                        });
                    
                        std::vector<FreeRegion> mergedRegions;
                        mergedRegions.push_back(freeRegions[0]);
                        
                        for (size_t i = 1; i < freeRegions.size(); i++) {
                            FreeRegion& last = mergedRegions.back();
                            const FreeRegion& current = freeRegions[i];
                            
                            if (last.offset + last.size == current.offset) {
                                // Merge adjacent regions
                                last.size += current.size;
                            } else {
                                mergedRegions.push_back(current);
                            }
                        }
                        
                        freeRegions = std::move(mergedRegions);
                    }
                    
                    // Try to find a free region big enough
                    bool allocated = false;
                    for (auto it = freeRegions.begin(); it != freeRegions.end(); ++it) {
                        if (it->size >= tensor->size) {
                            // Found a region big enough
                            size_t allocOffset = it->offset;
                            
                            // Update the free region
                            it->offset += tensor->size;
                            it->size -= tensor->size;
                            
                            // Remove the region if it's now empty
                            if (it->size == 0) {
                                freeRegions.erase(it);
                            }
                            
                            // Allocate the tensor
                            plan.allocateTensor(tensor, mainPoolIdx, allocOffset);
                            allocatedTensors.push_back({tensor, allocOffset});
                            allocated = true;
                            break;
                        }
                    }
                    
                    // For handling the "pool expansion" scenario:
                    if (!allocated) {
                        // Get the current pool size
                        size_t poolSize = plan.getMemoryPools()[mainPoolIdx].size;
                        
                        // Calculate new size needed
                        size_t newSize = poolSize + std::max(tensor->size, (size_t)1024 * 1024);
                        
                        // Create a new free region at the end of the current pool
                        FreeRegion newRegion = {poolSize, newSize - poolSize};
                        freeRegions.push_back(newRegion);
                        
                        // Resize the pool using a proper method
                        plan.resizePool(mainPoolIdx, newSize);
                        
                        // Try to allocate the tensor in the newly created region
                        size_t allocOffset = poolSize;  // Place at start of new region
                        
                        // Add the allocation
                        plan.allocateTensor(tensor, mainPoolIdx, allocOffset);
                        allocatedTensors.push_back({tensor, allocOffset});
                        
                        // Update the free regions list (remove the one we just used)
                        freeRegions.pop_back();
                        
                        // If there's space left after allocation, add as free region
                        if (poolSize + tensor->size < newSize) {
                            freeRegions.push_back({poolSize + tensor->size, newSize - (poolSize + tensor->size)});
                        }
                    }
                }
                
                // Compute memory usage timeline
                int maxStep = findMaxExecutionStep(tensors);
                computeMemoryUsageTimeline(tensors, maxStep);
                
                // Compute final metrics
                computeMetrics(tensors, plan);
            });
            
            return plan;
        }
    };
} // namespace strategies
} // namespace optimizer

namespace {
    static bool greedy_first_fit_registered = 
        optimizer::registry::registerOptimizer<optimizer::strategies::GreedyFirstFit>(
            "greedy-first-fit"
        );
}