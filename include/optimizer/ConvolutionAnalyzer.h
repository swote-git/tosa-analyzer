#ifndef CONVOLUTION_ANALYZER_H
#define CONVOLUTION_ANALYZER_H

#include "optimizer/OperationOptimizerInterface.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace optimizer {

class ConvolutionAnalyzer {
public:
    ConvolutionAnalyzer(TensorGraph* graph);
    
    // Analyze a specific conv2d operation
    ConvolutionOptimizationInfo analyzeConvolution(mlir::tosa::Conv2DOp convOp);
    
    // Check if Winograd algorithm would be beneficial
    bool isWinogradBeneficial(mlir::tosa::Conv2DOp convOp);
    
    // Check if im2col transformation would be beneficial
    bool isIm2ColBeneficial(mlir::tosa::Conv2DOp convOp);
    
    // Determine optimal tiling strategy
    int determineBestTileSize(mlir::tosa::Conv2DOp convOp);
    
    // Determine best memory layout for this hardware
    std::string determineBestLayout(mlir::tosa::Conv2DOp convOp);
    
private:
    TensorGraph* graph;
    
    // Helper method to extract tensor dimensions
    std::vector<int64_t> getConvolutionDimensions(mlir::tosa::Conv2DOp convOp);
    
    // Helper to check if shapes are compile-time constants
    bool hasConstantShapes(mlir::tosa::Conv2DOp convOp);
};

} // namespace optimizer

#endif // CONVOLUTION_ANALYZER_H