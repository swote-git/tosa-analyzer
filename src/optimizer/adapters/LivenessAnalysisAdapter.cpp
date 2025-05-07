// src/optimizer/adapters/LivenessAnalysisAdapter.cpp
#include "LivenessAnalysisAdapter.h"
#include "mlir/IR/BuiltinTypes.h"

namespace optimizer {
namespace adapters {

std::vector<TensorLifetimeInfo> LivenessAnalysisAdapter::convertToTensorLifetimeInfo(
    LivenessAnalysis* liveness, TensorGraph* graph) {
    
    std::vector<TensorLifetimeInfo> result;
    
    // Convert from liveRanges in LivenessAnalysis to TensorLifetimeInfo
    for (auto& entry : liveness->liveRanges) {
        mlir::Value value = entry.getFirst();
        auto& range = entry.getSecond();
        
        // Process only tensor values (skip non-tensor values)
        if (!mlir::isa<mlir::ShapedType>(value.getType()))
            continue;
        
        // Find the indices of creation and last use points
        int defPoint = -1;
        int lastUsePoint = -1;
        
        if (range.defNode) {
            for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                if (liveness->topoSortedNodes[i] == range.defNode) {
                    defPoint = i;
                    break;
                }
            }
        }
        
        if (range.lastUseNode) {
            for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                if (liveness->topoSortedNodes[i] == range.lastUseNode) {
                    lastUsePoint = i;
                    break;
                }
            }
        }
        
        // Generate ID (unique identifier needed)
        std::string id = "tensor_" + std::to_string(result.size());
        
        // Determine if model input or output
        bool isModelInput = (range.defNode == nullptr); // No defining node means it's an input
        bool isModelOutput = true; // Default assumption is output
        
        // If this value has any users, it's not an output
        if (!range.useNodes.empty()) {
            isModelOutput = false;
        }
        
        // If it's the result of the last node and has no users, it's an output
        if (range.defNode && range.useNodes.empty()) {
            isModelOutput = true;
        }
        
        // Create and add TensorLifetimeInfo
        TensorLifetimeInfo info(
            id, value, estimateTensorSize(value),
            defPoint, lastUsePoint, isModelInput, isModelOutput
        );
        
        result.push_back(info);
    }
    
    return result;
}

size_t LivenessAnalysisAdapter::estimateTensorSize(mlir::Value tensor) {
    // Copied from MemoryPlanner::estimateTensorSize function
    auto type = mlir::dyn_cast<mlir::ShapedType>(tensor.getType());
    if (!type) {
        return 0; // Not a tensor
    }
    
    // Calculate number of elements
    int64_t numElements = 1;
    for (auto dim : type.getShape()) {
        if (dim < 0) {
            // Dynamic dimension, use a default size
            dim = 1;
        }
        numElements *= dim;
    }
    
    // Determine element size based on element type
    size_t elementSize = 4; // Default to float32 (4 bytes)
    
    if (type.getElementType().isIntOrIndex()) {
        // Integer type
        unsigned bitWidth = 32; // Default to 32-bit
        
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type.getElementType())) {
            bitWidth = intType.getWidth();
        }
        
        elementSize = (bitWidth + 7) / 8; // Round up to nearest byte
    }
    else if (type.getElementType().isF16()) {
        elementSize = 2; // 16-bit float
    }
    else if (type.getElementType().isF64()) {
        elementSize = 8; // 64-bit float
    }
    
    return numElements * elementSize;
}

} // namespace adapters
} // namespace optimizer