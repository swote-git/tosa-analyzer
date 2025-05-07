#ifndef LIVENESS_ANALYSIS_ADAPTER_H
#define LIVENESS_ANALYSIS_ADAPTER_H

#include "../../LivenessAnalysis.h"
#include "../TensorLifetimeInfo.h"
#include <vector>

namespace optimizer {
namespace adapters {

class LivenessAnalysisAdapter {
public:
    // convert from LivenessAnalysis
    static std::vector<TensorLifetimeInfo> convertToTensorLifetimeInfo(
        LivenessAnalysis* liveness, TensorGraph* graph);
    
private:
    // estimate tensorSize
    static size_t estimateTensorSize(mlir::Value tensor);
};

} // namespace adapters
} // namespace optimizer

#endif // LIVENESS_ANALYSIS_ADAPTER_H