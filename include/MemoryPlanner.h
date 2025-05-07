#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

#include "LivenessAnalysis.h"
#include "TensorGraph.h"
#include "TensorNode.h"
#include "mlir/IR/Value.h"
#include <cstddef>


class MemoryPlanner {
public:
    // memory Allocation record
    struct AllocationInfo {
        mlir::Value value;
        TensorNode* defNode;
        TensorNode* lastUseNode;
        size_t size;
        int allocatedPoolIndex;
        int offset;
        bool isModelInput;
        bool isModelOutput;
    };
private:
    TensorGraph* graph;
    LivenessAnalysis* liveness;

    // Memory allocation plan
    std::vector<AllocationInfo> allocations;

    struct MemoryPool {
        size_t size;
        std::string name;
        std::vector<std::pair<int, int>> freeIntervals; // (start, end) offsets
    };
    std::vector<MemoryPool> memoryPools;

    size_t totalMemory;
    size_t peakMemory;
    std::vector<size_t> memoryUsageTimeline;
    
    size_t estimateTensorSize(mlir::Value tensor);
    int getNodeIndex(TensorNode* node);
    int findMemoryPool(AllocationInfo& alloc);
    void insertIntoMemoryPool(int poolIndex, AllocationInfo& alloc);
    void buildMemoryUsageTimeline();

    int createNewMemoryPool(size_t initialSize);
    void compactMemoryPools();
    
public:
    MemoryPlanner(TensorGraph* graph, LivenessAnalysis* liveness);
    ~MemoryPlanner();

    // memory planning
    void computeTensorSizes();
    void buildAllocationPlan();
    void performMemoryOptimizer();
    void generateAllocationCode(const std::string& filename);

    // analysis and visualization results. 
    void printMemoryStatistics();
    void visualizeMemoryUsage(const std::string& filename);
    size_t getTotalMemoryUsage() const;
    size_t getPeakMemoryUsage() const;
};


#endif // MEMORY_PLANNER_H