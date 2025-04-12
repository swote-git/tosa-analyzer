#ifndef MEMORY_PLANNER_H
#define MEMORY_PLANNER_H

#include "LivenessAnalysis.h"
#include "TensorGrpah.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"


class MemmoryPlanner {
public:
    
    MemmoryPlanner(LivenessAnalysis);
    ~MemmoryPlanner();
};
#endif // MEMORY_PLANNER_H