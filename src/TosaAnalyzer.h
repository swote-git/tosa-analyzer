#ifndef TOSA_ANALYZER_H
#define TOSA_ANALYZER_H
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "TensorGraph.h"
#include "LivenessAnalysis.h"



class TosaAnalyzer {
private:
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    TensorGraph* graph;
    std::string inputFileName;
    std::string outputFileName;
    LivenessAnalysis* liveness;

public:
    TosaAnalyzer();
    ~TosaAnalyzer();

    bool parseCommandLine(int argc, char **argv);
    bool loadAndParseInputFile();
    TensorGraph* buildTensorGraph();
    void exportGraphToDot(const std::string& filename);
    void performLivenessAnalysis();
    void printResults();
    int run(int argc, char **argv);
};

#endif // TOSA_ANALYZER_H