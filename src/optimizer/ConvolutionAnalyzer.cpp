#include "optimizer/ConvolutionAnalyzer.h"
#include "mlir/IR/BuiltinTypes.h"

namespace optimizer {

ConvolutionAnalyzer::ConvolutionAnalyzer(TensorGraph* graph) : graph(graph) {
}

ConvolutionOptimizationInfo ConvolutionAnalyzer::analyzeConvolution(mlir::tosa::Conv2DOp convOp) {
    ConvolutionOptimizationInfo info;
    info.operation = convOp;
    
    // Can only optimize operations with constant shapes
    if (!hasConstantShapes(convOp)) {
        return info; // No optimizations possible
    }
    
    // Get dimensions
    auto dims = getConvolutionDimensions(convOp);
    
    // Check for Winograd applicability (usually good for 3x3 kernels)
    if (isWinogradBeneficial(convOp)) {
        info.useWinograd = true;
        info.estimatedSpeedup *= 2.25; // Typical speedup for 3x3 convs
    }
    
    // Check for im2col transformation (often good for large filters)
    if (isIm2ColBeneficial(convOp)) {
        info.useIm2Col = true;
        info.estimatedSpeedup *= 1.3; // Typical speedup for large filters
    }
    
    // Determine optimal tiling
    info.tileSize = determineBestTileSize(convOp);
    if (info.tileSize > 0) {
        info.estimatedSpeedup *= 1.4; // Typical speedup from tiling
    }
    
    // Determine optimal memory layout
    info.preferredLayout = determineBestLayout(convOp);
    
    return info;
}

bool ConvolutionAnalyzer::hasConstantShapes(mlir::tosa::Conv2DOp convOp) {
    // Check if input and weight tensors have constant shapes
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getInput().getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getWeight().getType());
        
    if (!inputType || !weightType)
        return false;
        
    // Check for dynamic dimensions (indicated by -1 in MLIR)
    for (auto dim : inputType.getShape()) {
        if (dim < 0) return false;
    }
    
    for (auto dim : weightType.getShape()) {
        if (dim < 0) return false;
    }
    
    return true;
}

std::vector<int64_t> ConvolutionAnalyzer::getConvolutionDimensions(mlir::tosa::Conv2DOp convOp) {
    std::vector<int64_t> dims;
    
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getInput().getType());
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getWeight().getType());
        
    if (!inputType || !weightType)
        return dims;
        
    // Get input dimensions [N, H, W, C]
    auto inputShape = inputType.getShape();
    // Get weight dimensions [H, W, C_in, C_out]
    auto weightShape = weightType.getShape();
    
    // Extract all relevant dimensions
    dims.insert(dims.end(), inputShape.begin(), inputShape.end());
    dims.insert(dims.end(), weightShape.begin(), weightShape.end());
    
    return dims;
}

bool ConvolutionAnalyzer::isWinogradBeneficial(mlir::tosa::Conv2DOp convOp) {
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getWeight().getType());
    if (!weightType) return false;
    
    auto shape = weightType.getShape();
    
    // Winograd is most beneficial for 3x3 kernels with stride 1
    if (shape[0] == 3 && shape[1] == 3) {
        auto stride = convOp.getStride();
        if (stride.size() == 2 && stride[0] == 1 && stride[1] == 1) {
            return true;
        }
    }
    
    return false;
}

bool ConvolutionAnalyzer::isIm2ColBeneficial(mlir::tosa::Conv2DOp convOp) {
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getWeight().getType());
    if (!weightType) return false;
    
    auto shape = weightType.getShape();
    
    // Im2Col is beneficial for large kernels
    if (shape[0] > 3 || shape[1] > 3) {
        return true;
    }
    
    // Also beneficial for kernels with large output channels
    if (shape[3] > 512) {
        return true;
    }
    
    return false;
}

int ConvolutionAnalyzer::determineBestTileSize(mlir::tosa::Conv2DOp convOp) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getInput().getType());
    if (!inputType) return 0;
    
    auto shape = inputType.getShape();
    
    // Simple heuristic: use tile sizes that evenly divide the input dimensions
    // or that fit well in the cache
    
    // Assuming L1 cache is around 32KB
    constexpr int L1_CACHE_SIZE = 32 * 1024;
    
    // Try common tile sizes and find one that divides evenly
    for (int tile : {8, 16, 32, 64}) {
        if (shape[1] % tile == 0 && shape[2] % tile == 0) {
            // Check if the tile fits in cache
            int elementSize = 4; // Assuming float32
            int tileMemory = tile * tile * shape[3] * elementSize;
            
            if (tileMemory < L1_CACHE_SIZE / 2) {
                return tile;
            }
        }
    }
    
    // If no perfect tile found, return a default
    return 16;
}

std::string ConvolutionAnalyzer::determineBestLayout(mlir::tosa::Conv2DOp convOp) {
    // In practice, this would depend on the target hardware
    // For most modern GPUs and accelerators, NHWC tends to be better
    // For some CPUs, NCHW might be better
    
    // Simple logic: use NCHW for small channel counts, otherwise NHWC
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(
        convOp.getInput().getType());
    if (!inputType) return "NHWC"; // Default
    
    auto shape = inputType.getShape();
    if (shape.size() != 4) return "NHWC"; // Default if not 4D
    
    int channels = shape[3]; // Assuming NHWC
    
    if (channels <= 4) {
        return "NCHW"; // Might be better for few channels
    } else {
        return "NHWC"; // Better for many channels
    }
}

} // namespace optimizer