#ifndef OPERATION_OTPIMIZER_INTERFACE_H
#define OPERATION_OTPIMIZER_INTERFACE_H

#include "mlir/IR/Operation.h"
#include "TensorGraph.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace optimizer {

struct ConvolutionOptimizationInfo {
    // Original operation details
    mlir::Operation* operation;
    // Optimization parameters
    bool useWinograd = false;
    bool useIm2Col = false;
    int tileSize = 0;
    std::string preferredLayout = "NHWC"; // or "NCHW"
    // Performance estimates
    float estimatedSpeedup = 1.0;
    // Additional metadata
    std::map<std::string, std::string> metadata;
};

class OperationOptimizerInterface {
public:
    virtual ~OperationOptimizerInterface() = default;
    
    // Main optimization method
    virtual void optimizeOperations(TensorGraph* graph, mlir::ModuleOp module) = 0;
    
    // Get name and description
    virtual std::string getName() const = 0;
    virtual std::string getDescription() const = 0;
};

} // namespace optimizer

#endif // OPERATION_OTPIMIZER_INTERFACE_H