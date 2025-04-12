#ifndef TENSOR_NODE_H
#define TENSOR_NODE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <string>
#include <vector>

class TensorNode {
public:
    // 노드 ID (디버깅 용도)
    std::string id;
    
    // 해당 MLIR 연산
    mlir::Operation* operation;
    
    // 입력값 (이전 노드들로부터 오는 에지)
    std::vector<mlir::Value> inputs;
    
    // 출력값 (다음 노드들로 가는 에지)
    std::vector<mlir::Value> outputs;
    
    // 연산 타입과 속성을 저장하는 추가 필드
    std::string opName;          // 연산 이름 (예: "tosa.add")
    std::string opTypeName;      // 다이얼렉트 이름 (예: "tosa")
    
    // 연산 속성을 문자열 맵으로 저장
    std::map<std::string, std::string> attributes;
    
    // 생성자
    TensorNode(mlir::Operation* op);
    
    // 연산 정보 출력
    void printOpInfo() const;
    
    // 특정 속성 값 가져오기
    std::string getAttribute(const std::string& name) const;
    
    // 이 연산이 특정 다이얼렉트에 속하는지 확인
    bool isDialect(const std::string& dialectName) const;
    
    // 이 연산이 특정 유형인지 확인
    bool isOpType(const std::string& opTypeName) const;
};

#endif // TENSOR_NODE_H