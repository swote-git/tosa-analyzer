#include "TensorNode.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

TensorNode::TensorNode(mlir::Operation* op) : operation(op) {
    // create node id
    id = op->getName().getStringRef().str() + "_" + 
         std::to_string(reinterpret_cast<uintptr_t>(op));
    
    // store operation name and dialect
    opName = op->getName().getStringRef().str();
    
    // extract dialect name (ex : "tosa.add" -> "tosa")
    if (op->getDialect()) {
        opTypeName = op->getDialect()->getNamespace().str();
    } else {
        // if no dialect, extract from operation name
        size_t dotPos = opName.find('.');
        if (dotPos != std::string::npos) {
            opTypeName = opName.substr(0, dotPos);
        }
    }
    
    // initialize input/output value
    for (auto operand : op->getOperands()) {
        inputs.push_back(operand);
    }
    
    for (auto result : op->getResults()) {
        outputs.push_back(result);
    }
    
    // store operation attributes
    for (auto attr : op->getAttrs()) {
        std::string attrName = attr.getName().str();
        std::string attrValue;
        
        // convert attributes value to string
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
    
    // print input info
    llvm::outs() << "Inputs (" << inputs.size() << "):\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        llvm::outs() << "  [" << i << "] " << input << " : " << input.getType() << "\n";
    }
    
    // print output info
    llvm::outs() << "Outputs (" << outputs.size() << "):\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        llvm::outs() << "  [" << i << "] " << output << " : " << output.getType() << "\n";
    }
    
    // print attributes info
    llvm::outs() << "Attributes (" << attributes.size() << "):\n";
    for (const auto& attr : attributes) {
        llvm::outs() << "  " << attr.first << " = " << attr.second << "\n";
    }
}