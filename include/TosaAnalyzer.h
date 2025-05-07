#ifndef TOSA_ANALYZER_H
#define TOSA_ANALYZER_H
#include "MemoryPlanner.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "TensorGraph.h"
#include "LivenessAnalysis.h"
#include <string>



class TosaAnalyzer {
private:
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    TensorGraph* graph;
    LivenessAnalysis* liveness;
    MemoryPlanner* memoryPlanner;

    std::string inputFileName;
    std::string livenessVisualFile;
    std::string memoryPlanFile;
    std::string memoryVisualFile;
public:
    TosaAnalyzer();
    ~TosaAnalyzer();

    bool parseCommandLine(int argc, char **argv);
    bool loadAndParseInputFile();
    TensorGraph* buildTensorGraph();
    void exportGraphToDot(const std::string& filename);
    void performLivenessAnalysis();
    void planMemoryAllocation();
    void printResults();
    int run(int argc, char **argv);
};

#endif // TOSA_ANALYZER_H