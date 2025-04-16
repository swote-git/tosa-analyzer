#ifndef TENSOR_NODE_H
#define TENSOR_NODE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <string>
#include <vector>

class TensorNode {
public:
    // node ID (for debug)
    std::string id;
    
    // MLIR operation
    mlir::Operation* operation;
    
    // input parameters values (incoimg edges)
    std::vector<mlir::Value> inputs;
    
    // output values (outgoing edges)
    std::vector<mlir::Value> outputs;
    
    // additional fields that store operation name and diaclect
    std::string opName;          // operation name (ex: "tosa.add")
    std::string opTypeName;      // dialect name (ex: "tosa")
    
    // store operation's attributes to map
    std::map<std::string, std::string> attributes;
    
    // constructor
    TensorNode(mlir::Operation* op);
    
    // print operation info
    void printOpInfo() const;
    
    // get attribute
    std::string getAttribute(const std::string& name) const;
    
    // Check if this operation belongs to a specific dialect
    bool isDialect(const std::string& dialectName) const;
    
    // Check if this operation is of a specific type
    bool isOpType(const std::string& opTypeName) const;
};

#endif // TENSOR_NODE_H