#include "MemoryOptimizerInterface.h"

namespace optimizer {

// Base class for memory optimizers with common functionality
class BaseMemoryOptimizer : public MemoryOptimizerInterface {
public:
    const OptimizationMetrics& getMetrics() const override {
        return metrics;
    }
    
protected:
    // Utility method to compute memory usage timeline
    void computeMemoryUsageTimeline(
        const std::vector<TensorLifetimeInfo>& tensors,
        int executionSteps) {
        
        // Initialize timeline with zeros
        metrics.memoryUsageTimeline.resize(executionSteps + 1, 0);
        
        // Add each tensor's memory to the timeline during its lifetime
        for (const auto& tensor : tensors) {
            for (int step = tensor.defPoint; step <= tensor.lastUsePoint; step++) {
                metrics.memoryUsageTimeline[step] += tensor.size;
            }
        }
        
        // Find peak memory usage
        metrics.peakMemoryUsage = 0;
        for (size_t usage : metrics.memoryUsageTimeline) {
            if (usage > metrics.peakMemoryUsage) {
                metrics.peakMemoryUsage = usage;
            }
        }
    }
    
    // Helper to find max execution step
    int findMaxExecutionStep(const std::vector<TensorLifetimeInfo>& tensors) {
        int maxStep = 0;
        for (const auto& tensor : tensors) {
            if (tensor.lastUsePoint > maxStep) {
                maxStep = tensor.lastUsePoint;
            }
        }
        return maxStep;
    }
    
    // Create a default pool for model inputs/outputs
    int createInputOutputPool(
        MemoryAllocationPlan& plan,
        const std::vector<TensorLifetimeInfo>& tensors) {
        
        // Estimate size needed for inputs/outputs
        size_t ioPoolSize = 0;
        for (const auto& tensor : tensors) {
            if (tensor.isModelInput || tensor.isModelOutput) {
                ioPoolSize += tensor.size;
            }
        }
        
        // Add some buffer
        ioPoolSize = std::max(ioPoolSize, (size_t)1024 * 1024);
        
        // Create the IO pool
        return plan.addMemoryPool("io_pool", ioPoolSize);
    }
};

} // namespace optimizer