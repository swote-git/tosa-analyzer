#ifndef OPTIMIZATION_METRICS_H
#define OPTIMIZATION_METRICS_H

#include <string>
#include <map>
#include <vector>

namespace optimizer {

// Metrics for evaluating memory optimization performance
struct OptimizationMetrics {
    // Basic memory metrics
    size_t totalTensorMemory;     // Sum of all tensor sizes
    size_t peakMemoryUsage;       // Maximum memory used at any point
    size_t totalPoolMemory;       // Total memory allocated across all pools
    
    // Derived efficiency metrics
    double temporalEfficiency;    // totalTensorMemory / peakMemoryUsage (%)
    double spatialEfficiency;     // totalTensorMemory / totalPoolMemory (%)
    
    // Performance metrics
    double optimizationTimeMs;    // Time taken to compute allocation plan
    
    // Advanced metrics
    int numberOfPools;            // Number of memory pools used
    size_t largestPoolSize;       // Size of the largest memory pool
    double fragmentationRatio;    // Unused memory / total pool memory (%)
    
    // Timeline metrics
    std::vector<size_t> memoryUsageTimeline;  // Memory usage at each execution step
    
    // Create a string representation of all metrics
    std::string toString() const;
    
    // Support for custom metrics
    std::map<std::string, double> customMetrics;
};

} // namespace optimizer

#endif // OPTIMIZATION_METRICS_H