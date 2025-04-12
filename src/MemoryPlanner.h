#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

#include "LivenessAnalysis.h"
#include "TensorGraph.h"
#include "TensorNode.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"


class MemmoryPlanner {
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
public:

    MemmoryPlanner(TensorGraph* tensorgraph, LivenessAnalysis* livness);
    ~MemmoryPlanner();

    // memory planning
    void computeTensorSize();
    void buildAllocationPlan();
    void memoryOptimizer();
    void generateAllocationCode(const std::string&);

    // analysis and visualization results. 
    void printMemoryStatistics();
    void visualizationMemoryUsage();
    size_t getTotalMemoryUsage();
    size_t getPeakMemoryUsage();
};
#endif // MEMORY_PLANNER_H