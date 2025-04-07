#pragma once
#include <vector>
#include <string>
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"


// 그래프의 노드를 표현하는 클래스
class TensorNode {
public:
    TensorNode(mlir::Operation* op);
    // 노드 ID (디버깅 용도)
    std::string id;
    
    // 해당 MLIR 연산
    mlir::Operation* operation;
    
    // 입력값 (이전 노드들로부터 오는 에지)
    std::vector<mlir::Value> inputs;
    
    // 출력값 (다음 노드들로 가는 에지)
    std::vector<mlir::Value> outputs;
    
};