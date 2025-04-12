#ifndef TOSA_ANALYZER_H
#define TOSA_ANALYZER_H
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/InitLLVM.h"
#include "TensorNode.h"
#include "TensorGraph.h"
#include "LivenessAnalysis.h"



class TosaAnalyzer {
private:
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    TensorGraph* graph;
    std::string inputFileName;
    std::string outputFileName;
    LivenessAnalysis* livenessAnalysis;

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