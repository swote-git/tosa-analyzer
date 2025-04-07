#include "TensorNode.h"

// TensorNode Expression
TensorNode::TensorNode(mlir::Operation* op) : operation(op) {
    // 노드 ID 생성 (예: 연산 이름 + 고유 식별자)
    id = op->getName().getStringRef().str() + "_" + 
            std::to_string(reinterpret_cast<uintptr_t>(op));
    
    // 입력 및 출력 값 초기화
    for (auto operand : op->getOperands()) {
        inputs.push_back(operand);
    }
    
    for (auto result : op->getResults()) {
        outputs.push_back(result);
    }
}