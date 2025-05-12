#include "BaseMemoryOptimizer.cpp"
#include "optimizer/registry/OptimizerRegistry.h"

namespace optimizer {
namespace strategies {

class PeakMinimizingPlanner : public BaseMemoryOptimizer {
public:
    std::string getName() const override {
        return "PeakMinimizingPlanner";
    }
    
    std::string getDescription() const override {
        return "Memory planner that minimizes peak memory usage through interval-based allocation";
    }
    
    MemoryAllocationPlan optimize(const std::vector<TensorLifetimeInfo>& tensors) override {
        MemoryAllocationPlan plan;
        
        measureExecutionTime([&]() {
            // Create pool for inputs and outputs
            int ioPoolIdx = createInputOutputPool(plan, tensors);
            
            // Create main working memory pool
            int mainPoolIdx = plan.addMemoryPool("main_pool", 1024 * 1024); // Initial size
            
            // Step 1: Compute memory consumption at each execution point
            int maxStep = findMaxExecutionStep(tensors);
            std::vector<size_t> memoryAtStep(maxStep + 1, 0);
            computeMemoryTimelineWithoutOptimization(tensors, memoryAtStep);
            size_t initialPeakMemory = *std::max_element(memoryAtStep.begin(), memoryAtStep.end());
            
            // Step 2: Sort tensors by definition time
            std::vector<const TensorLifetimeInfo*> tensorsByDefTime;
            for (const auto& tensor : tensors) {
                if (!tensor.isModelInput && !tensor.isModelOutput) {
                    tensorsByDefTime.push_back(&tensor);
                }
            }
            std::sort(tensorsByDefTime.begin(), tensorsByDefTime.end(),
                [](const TensorLifetimeInfo* a, const TensorLifetimeInfo* b) {
                    return a->defPoint < b->defPoint;
                });
            
            // Step 3: Allocate model inputs/outputs
            for (const auto& tensor : tensors) {
                if (tensor.isModelInput || tensor.isModelOutput) {
                    plan.allocateTensor(&tensor, ioPoolIdx, 0);
                }
            }
            
            // Step 4: For each tensor, find best fitting memory that's free
            // Track allocated regions and their lifetimes
            struct MemoryRegion {
                size_t offset;
                size_t size;
                int lastUsePoint;
            };
            std::vector<MemoryRegion> allocatedRegions;
            
            for (const auto* tensor : tensorsByDefTime) {
                // Find all regions that are free at the time this tensor is defined
                std::vector<MemoryRegion> availableRegions;
                
                // Check which already-allocated regions are now free
                for (const auto& region : allocatedRegions) {
                    if (region.lastUsePoint < tensor->defPoint) {
                        availableRegions.push_back(region);
                    }
                }
                
                // Sort available regions by size (best-fit approach)
                std::sort(availableRegions.begin(), availableRegions.end(),
                    [tensor](const MemoryRegion& a, const MemoryRegion& b) {
                        // If both regions are big enough, prefer the smaller one (less waste)
                        if (a.size >= tensor->size && b.size >= tensor->size)
                            return a.size < b.size;
                        // Otherwise, prefer the larger one (more likely to fit)
                        return a.size > b.size;
                    });
                
                if (!availableRegions.empty() && availableRegions[0].size >= tensor->size) {
                    // Found a region we can reuse
                    MemoryRegion selectedRegion = availableRegions[0];
                    
                    // Remove this region from the allocated list
                    allocatedRegions.erase(
                        std::remove_if(allocatedRegions.begin(), allocatedRegions.end(),
                            [&selectedRegion](const MemoryRegion& r) {
                                return r.offset == selectedRegion.offset && r.size == selectedRegion.size;
                            }),
                        allocatedRegions.end());
                    
                    // Allocate tensor here
                    plan.allocateTensor(tensor, mainPoolIdx, selectedRegion.offset);
                    
                    // Add back to allocated regions with new lifetime
                    selectedRegion.lastUsePoint = tensor->lastUsePoint;
                    allocatedRegions.push_back(selectedRegion);
                    
                    // If the region is larger than needed, split it
                    if (selectedRegion.size > tensor->size) {
                        MemoryRegion remainingRegion;
                        remainingRegion.offset = selectedRegion.offset + tensor->size;
                        remainingRegion.size = selectedRegion.size - tensor->size;
                        remainingRegion.lastUsePoint = selectedRegion.lastUsePoint;
                        allocatedRegions.push_back(remainingRegion);
                    }
                } else {
                    // Need to allocate new memory
                    // Find the current size of the pool
                    size_t currentOffset = 0;
                    for (const auto& region : allocatedRegions) {
                        size_t regionEnd = region.offset + region.size;
                        if (regionEnd > currentOffset) {
                            currentOffset = regionEnd;
                        }
                    }
                    
                    // Allocate at the end
                    plan.allocateTensor(tensor, mainPoolIdx, currentOffset);
                    
                    // Update the pool size if needed
                    size_t requiredSize = currentOffset + tensor->size;
                    if (requiredSize > plan.getMemoryPools()[mainPoolIdx].size) {
                        plan.resizePool(mainPoolIdx, requiredSize);
                    }
                    
                    // Add to allocated regions
                    MemoryRegion newRegion;
                    newRegion.offset = currentOffset;
                    newRegion.size = tensor->size;
                    newRegion.lastUsePoint = tensor->lastUsePoint;
                    allocatedRegions.push_back(newRegion);
                }
            }
            
            // Compute metrics after optimization
            computeMemoryUsageTimeline(tensors, maxStep);
            computeMetrics(tensors, plan);
            
            // Add a custom metric for memory reduction
            metrics.customMetrics["memory_reduction_percent"] = 
                100.0 * (1.0 - (double)metrics.peakMemoryUsage / initialPeakMemory);
        });
        
        return plan;
    }

private:
    void computeMemoryTimelineWithoutOptimization(
        const std::vector<TensorLifetimeInfo>& tensors,
        std::vector<size_t>& timeline) {
        
        for (const auto& tensor : tensors) {
            for (int step = tensor.defPoint; step <= tensor.lastUsePoint; step++) {
                timeline[step] += tensor.size;
            }
        }
    }
};

} // namespace strategies
} // namespace optimizer

namespace {
    static bool peak_minimizing_registered = 
        optimizer::registry::registerOptimizer<optimizer::strategies::PeakMinimizingPlanner>(
            "peak-minimizing"
        );
}