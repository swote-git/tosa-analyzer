#ifndef TENSOR_LIFETIME_INFO_H
#define TENSOR_LIFETIME_INFO_H

#include "mlir/IR/Value.h"
#include <string>

namespace optimizer {

// Represents the complete lifetime information of a tensor
struct TensorLifetimeInfo {
    // Unique identifier for the tensor
    std::string id;
    
    // MLIR value representing this tensor
    mlir::Value value;
    
    // Size in bytes
    size_t size;
    
    // Lifetime boundaries (execution index in topological order)
    int defPoint;
    int lastUsePoint;
    
    // Is this a model input or output (special handling)
    bool isModelInput;
    bool isModelOutput;
    
    // Type information
    mlir::Type type;
    
    // Helper constructor
    TensorLifetimeInfo(
        const std::string& id,
        mlir::Value value,
        size_t size,
        int defPoint,
        int lastUsePoint,
        bool isInput = false,
        bool isOutput = false
    ) : id(id), value(value), size(size), defPoint(defPoint), 
        lastUsePoint(lastUsePoint), isModelInput(isInput), 
        isModelOutput(isOutput), type(value.getType()) {}
    
    // Check if this tensor's lifetime overlaps with another tensor
    bool overlaps(const TensorLifetimeInfo& other) const {
        return !(lastUsePoint < other.defPoint || other.lastUsePoint < defPoint);
    }
};

} // namespace optimizer

#endif // TENSOR_LIFETIME_INFO_H