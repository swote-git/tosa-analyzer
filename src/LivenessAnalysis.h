#ifndef LIVENESS_ANALYSIS_H
#define LIVENESS_ANALYSIS_H

#include "TensorGraph.h"
#include "TensorNode.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

class LivenessAnalysis {
public:
    // Use DenseMap instead of unordered_map for node liveness information
    llvm::DenseMap<TensorNode*, llvm::DenseSet<mlir::Value>> liveIn;
    llvm::DenseMap<TensorNode*, llvm::DenseSet<mlir::Value>> liveOut;
    
    // Define the LiveRange structure as before
    struct LiveRange {
        TensorNode* defNode;
        std::vector<TensorNode*> useNodes;
        TensorNode* lastUseNode; // Based on topological sort order
    };
    
    // Use DenseMap for liveRanges too
    llvm::DenseMap<mlir::Value, LiveRange> liveRanges;
    
    // Topologically sorted nodes (execution order)
    std::vector<TensorNode*> topoSortedNodes;
    
    // Constructor: perform liveness analysis on the data flow graph
    LivenessAnalysis(TensorGraph* graph);

    // topological sort (DFS base)
    void topologicalSort(TensorGraph* graph);

    void topoSortUtil(TensorNode* node, std::set<TensorNode*>& visited, 
                        std::set<TensorNode*>& temp, TensorGraph* graph);
    
    // bulid definition node and usage node of each value tensor
    void buildDefUseInfo(TensorGraph* graph);
    
    // compute liveness analysis (in reverse order of topological sort results)
    void computeLiveness();
    
    // compute live range of each tensor
    void computeLiveRanges();
    
    // print liveness info
    void printLivenessInfo();
};

#endif
