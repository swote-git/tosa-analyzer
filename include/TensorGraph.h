#ifndef TENSOR_GRAPH_H
#define TENSOR_GRAPH_H
#include "TensorNode.h"
#include "llvm/ADT/DenseMap.h"

// data-flow graph
class TensorGraph {
public:
    TensorGraph();
    ~TensorGraph();
    
    // store all node
    std::vector<TensorNode*> nodes;
    
    // map to find nodes that define Value
    llvm::DenseMap<mlir::Value, TensorNode*> definingNodes;
    
    // map to find nodes that using Value
    llvm::DenseMap<mlir::Value, std::vector<TensorNode*>> userNodes;
};
    
#endif