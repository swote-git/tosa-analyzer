#include "TensorNode.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"

TensorNode::TensorNode(mlir::Operation* op) : operation(op) {
    // 노드 ID 생성 (예: 연산 이름 + 고유 식별자)
    id = op->getName().getStringRef().str() + "_" + 
         std::to_string(reinterpret_cast<uintptr_t>(op));
    
    // 연산 이름과 다이얼렉트 이름 저장
    opName = op->getName().getStringRef().str();
    
    // 다이얼렉트 이름 추출 (예: "tosa.add" -> "tosa")
    if (op->getDialect()) {
        opTypeName = op->getDialect()->getNamespace().str();
    } else {
        // 다이얼렉트가 없으면 연산 이름에서 추출 시도
        size_t dotPos = opName.find('.');
        if (dotPos != std::string::npos) {
            opTypeName = opName.substr(0, dotPos);
        }
    }
    
    // 입력 및 출력 값 초기화
    for (auto operand : op->getOperands()) {
        inputs.push_back(operand);
    }
    
    for (auto result : op->getResults()) {
        outputs.push_back(result);
    }
    
    // 연산 속성 저장
    for (auto attr : op->getAttrs()) {
        std::string attrName = attr.getName().str();
        std::string attrValue;
        
        // 속성 값을 문자열로 변환
        llvm::raw_string_ostream stream(attrValue);
        attr.getValue().print(stream);
        
        attributes[attrName] = attrValue;
    }
}

std::string TensorNode::getAttribute(const std::string& name) const {
    auto it = attributes.find(name);
    if (it != attributes.end()) {
        return it->second;
    }
    return "";
}

bool TensorNode::isDialect(const std::string& dialectName) const {
    return opTypeName == dialectName;
}

bool TensorNode::isOpType(const std::string& opType) const {
    return opName == opType;
}

void TensorNode::printOpInfo() const {
    llvm::outs() << "Operation: " << opName << "\n";
    llvm::outs() << "Dialect: " << opTypeName << "\n";
    
    // 입력값 정보 출력
    llvm::outs() << "Inputs (" << inputs.size() << "):\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        llvm::outs() << "  [" << i << "] " << input << " : " << input.getType() << "\n";
    }
    
    // 출력값 정보 출력
    llvm::outs() << "Outputs (" << outputs.size() << "):\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        llvm::outs() << "  [" << i << "] " << output << " : " << output.getType() << "\n";
    }
    
    // 속성 정보 출력
    llvm::outs() << "Attributes (" << attributes.size() << "):\n";
    for (const auto& attr : attributes) {
        llvm::outs() << "  " << attr.first << " = " << attr.second << "\n";
    }
}