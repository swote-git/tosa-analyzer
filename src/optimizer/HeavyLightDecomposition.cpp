#include "BaseMemoryOptimizer.cpp"
#include <numeric>

namespace optimizer {

class HeavyLightDecomposition : public BaseMemoryOptimizer {
public:
    std::string getName() const override {
        return "HeavyLightDecomposition";
    }
    
    std::string getDescription() const override {
        return "A bin-packing algorithm that separates tensors into 'heavy' and 'light' "
               "categories, optimizing them separately for improved memory utilization.";
    }
    
    MemoryAllocationPlan optimize(
        const std::vector<TensorLifetimeInfo>& tensors) override {
        
        MemoryAllocationPlan plan;
        
        // Measure execution time
        measureExecutionTime([&]() {
            // Create IO pool for inputs/outputs
            int ioPoolIdx = createInputOutputPool(plan, tensors);
            
            // Step 1: Calculate total tensor memory
            size_t totalMemory = 0;
            for (const auto& tensor : tensors) {
                totalMemory += tensor.size;
            }
            
            // Step 2: Classify tensors as "heavy" or "light"
            const float HEAVY_THRESHOLD = 0.1; // Tensors using >10% of total memory are "heavy"
            size_t sizeThreshold = totalMemory * HEAVY_THRESHOLD;
            
            std::vector<const TensorLifetimeInfo*> heavyTensors;
            std::vector<const TensorLifetimeInfo*> lightTensors;
            
            for (const auto& tensor : tensors) {
                // Skip inputs/outputs
                if (tensor.isModelInput || tensor.isModelOutput) {
                    plan.allocateTensor(&tensor, ioPoolIdx, 0);
                    continue;
                }
                
                // Classify as heavy or light
                if (tensor.size >= sizeThreshold) {
                    heavyTensors.push_back(&tensor);
                } else {
                    lightTensors.push_back(&tensor);
                }
            }
            
            // Step 3: Sort heavy tensors by size (largest first)
            std::sort(heavyTensors.begin(), heavyTensors.end(), 
                [](const TensorLifetimeInfo* a, const TensorLifetimeInfo* b) {
                    return a->size > b->size;
                });
            
            // Step 4: Sort light tensors by lifetime (earliest definition first)
            std::sort(lightTensors.begin(), lightTensors.end(), 
                [](const TensorLifetimeInfo* a, const TensorLifetimeInfo* b) {
                    return a->defPoint < b->defPoint;
                });
            
            // Step 5: Allocate heavy tensors first (each in its own dedicated pool)
            for (auto heavyTensor : heavyTensors) {
                int poolIdx = plan.addMemoryPool(
                    "heavy_pool_" + heavyTensor->id,
                    heavyTensor->size
                );
                plan.allocateTensor(heavyTensor, poolIdx, 0);
            }
            
            // Step 6: Create a shared pool for light tensors
            int lightPoolIdx = plan.addMemoryPool("light_pool", totalMemory / 2);
            
            // Data structure for tracking memory regions and lifetimes
            struct MemoryRegion {
                size_t offset;
                size_t size;
                int lastUsePoint;  // When this region becomes free again
            };
            
            std::vector<MemoryRegion> allocatedRegions;
            size_t lightPoolSize = totalMemory / 2;
            size_t nextFreeOffset = 0;
            
            // Step 7: Allocate light tensors with lifetime-aware best-fit
            for (auto lightTensor : lightTensors) {
                // First, free any memory regions that are no longer needed
                for (auto it = allocatedRegions.begin(); it != allocatedRegions.end();) {
                    if (it->lastUsePoint < lightTensor->defPoint) {
                        // This region is free now, remove it
                        it = allocatedRegions.erase(it);
                    } else {
                        ++it;
                    }
                }
                
                // Find best-fit region from free space
                bool allocated = false;
                size_t bestFitOffset = 0;
                size_t bestFitSize = SIZE_MAX;
                
                // Track allocated offsets to find gaps
                std::vector<std::pair<size_t, size_t>> allocatedOffsets;
                for (const auto& region : allocatedRegions) {
                    allocatedOffsets.push_back({region.offset, region.offset + region.size});
                }
                
                // Sort by offset
                std::sort(allocatedOffsets.begin(), allocatedOffsets.end());
                
                // Find gaps between allocated regions
                size_t currentOffset = 0;
                for (const auto& region : allocatedOffsets) {
                    if (region.first > currentOffset) {
                        // There's a gap here
                        size_t gapSize = region.first - currentOffset;
                        
                        if (gapSize >= lightTensor->size && gapSize < bestFitSize) {
                            // This is a better fit
                            bestFitOffset = currentOffset;
                            bestFitSize = gapSize;
                            allocated = true;
                        }
                    }
                    currentOffset = std::max(currentOffset, region.second);
                }
                
                // If we couldn't find a suitable gap, allocate at the end
                if (!allocated) {
                    // Check if we need to expand the pool
                    if (nextFreeOffset + lightTensor->size > lightPoolSize) {
                        // Double the pool size
                        lightPoolSize *= 2;
                        // Update the pool in the plan
                        auto& pools = const_cast<std::vector<MemoryPool>&>(plan.getMemoryPools());
                        pools[lightPoolIdx].size = lightPoolSize;
                    }
                    
                    bestFitOffset = nextFreeOffset;
                    nextFreeOffset += lightTensor->size;
                }
                
                // Allocate the tensor
                plan.allocateTensor(lightTensor, lightPoolIdx, bestFitOffset);
                
                // Track this allocation
                allocatedRegions.push_back({
                    bestFitOffset,
                    lightTensor->size,
                    lightTensor->lastUsePoint
                });
            }
            
            // Compute memory usage timeline
            int maxStep = findMaxExecutionStep(tensors);
            computeMemoryUsageTimeline(tensors, maxStep);
            
            // Compute final metrics
            computeMetrics(tensors, plan);
            
            // Add custom metrics specific to this algorithm
            metrics.customMetrics["heavy_tensors_count"] = heavyTensors.size();
            metrics.customMetrics["light_tensors_count"] = lightTensors.size();
            metrics.customMetrics["heavy_memory_percentage"] = 
                (totalMemory > 0) ? 
                (double)std::accumulate(heavyTensors.begin(), heavyTensors.end(), 0ULL,
                    [](size_t sum, const TensorLifetimeInfo* t) { return sum + t->size; }) 
                / totalMemory * 100.0 : 0.0;
        });
        
        return plan;
    }
};

} // namespace optimizer