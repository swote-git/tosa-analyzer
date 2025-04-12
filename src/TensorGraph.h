#ifndef TENSOR_GRAPH_H
#define TENSOR_GRAPH_H
#include "TensorNode.h"
#include "llvm/ADT/DenseMap.h"

// 전체 데이터 흐름 그래프
class TensorGraph {
public:
    TensorGraph();
    // 소멸자: 노드 메모리 정리
    ~TensorGraph();
    
    // 모든 노드들을 저장
    std::vector<TensorNode*> nodes;
    
    // Value를 정의하는 노드를 빠르게 찾기 위한 맵
    llvm::DenseMap<mlir::Value, TensorNode*> definingNodes;
    // Value를 사용하는 노드들을 빠르게 찾기 위한 맵
    llvm::DenseMap<mlir::Value, std::vector<TensorNode*>> userNodes;
};
    
#endif