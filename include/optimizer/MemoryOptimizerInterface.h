#ifndef MEMORY_OPTIMIZER_INTERFACE_H
#define MEMORY_OPTIMIZER_INTERFACE_H

#include "MemoryAllocationPlan.h"
#include "OptimizationMetrics.h"
#include "TensorLifetimeInfo.h"
#include <vector>
#include <string>
#include <chrono>

namespace optimizer {

// Abstract interface for memory optimizers
class MemoryOptimizerInterface {
public:
    // Virtual destructor for proper cleanup
    virtual ~MemoryOptimizerInterface() = default;
    
    // Get the name of this optimizer
    virtual std::string getName() const = 0;
    
    // Get a description of the optimization strategy
    virtual std::string getDescription() const = 0;
    
    // Main optimization method
    // Takes tensor lifetime information and produces a memory allocation plan
    virtual MemoryAllocationPlan optimize(
        const std::vector<TensorLifetimeInfo>& tensors) = 0;
    
    // Get performance metrics from the last optimization run
    virtual const OptimizationMetrics& getMetrics() const = 0;
    
protected:
    // Helper method to measure optimization time
    void measureExecutionTime(const std::function<void()>& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        metrics.optimizationTimeMs = duration.count();
    }
    
    // Compute basic metrics after optimization
    void computeMetrics(
        const std::vector<TensorLifetimeInfo>& tensors,
        const MemoryAllocationPlan& plan) {
        
        // Calculate total tensor memory
        metrics.totalTensorMemory = 0;
        for (const auto& tensor : tensors) {
            metrics.totalTensorMemory += tensor.size;
        }
        
        // Get pool metrics
        metrics.numberOfPools = plan.getMemoryPools().size();
        metrics.totalPoolMemory = plan.getTotalMemoryUsage();
        
        // Find largest pool
        metrics.largestPoolSize = 0;
        for (const auto& pool : plan.getMemoryPools()) {
            if (pool.size > metrics.largestPoolSize) {
                metrics.largestPoolSize = pool.size;
            }
        }
        
        // Compute efficiency metrics
        if (metrics.peakMemoryUsage > 0) {
            metrics.temporalEfficiency = 
                (double)metrics.totalTensorMemory / metrics.peakMemoryUsage * 100.0;
        }
        
        if (metrics.totalPoolMemory > 0) {
            metrics.spatialEfficiency = 
                (double)metrics.totalTensorMemory / metrics.totalPoolMemory * 100.0;
            
            metrics.fragmentationRatio = 
                (double)(metrics.totalPoolMemory - metrics.totalTensorMemory) 
                / metrics.totalPoolMemory * 100.0;
        }
    }
    
    // Storage for metrics
    OptimizationMetrics metrics;
};

} // namespace optimizer

#endif // MEMORY_OPTIMIZER_INTERFACE_H