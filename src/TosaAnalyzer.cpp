#include "TosaAnalyzer.h"
#include "LivenessAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "TensorNode.h"

// constructor
TosaAnalyzer::TosaAnalyzer() : graph(nullptr), livenessAnalysis(nullptr) {
    // register dialect
    context.loadDialect<mlir::tosa::TosaDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
}

// Destructor
TosaAnalyzer::~TosaAnalyzer() {
    if (graph)
        delete graph;
    if (livenessAnalysis) 
        delete livenessAnalysis;
}

// parse input from command line
bool TosaAnalyzer::parseCommandLine(int argc, char **argv) {
    // LLVM initialization
    llvm::InitLLVM initLLVM(argc, argv);
    
    // command line options
    llvm::cl::opt<std::string> input(
        "input",
        llvm::cl::desc("Input MLIR file"),
        llvm::cl::Required);
    
    llvm::cl::opt<std::string> output(
        "output-dot",
        llvm::cl::desc("Output DOT file for graph visualization"),
        llvm::cl::init("tensorGrpah.dot"));
    
    llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TOSA Model Liveness Analyzer\n");

    // store file values
    inputFileName = input;
    outputFileName = output;

    return true;
}

// load and parse input file
bool TosaAnalyzer::loadAndParseInputFile() {
    // loading source file
    std::string errorMessage;
    auto file = mlir::openInputFile(inputFileName, &errorMessage);
    
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }
    
    // set up for source manager
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    
    // parse MLIR
    mlir::OwningOpRef<mlir::Operation*> moduleRef = 
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!moduleRef) {
        llvm::errs() << "Error parsing input file\n";
        return 1;
    }
    
    mlir::ModuleOp module = mlir::cast<mlir::ModuleOp>(moduleRef.get());
    
    // prints MLIR info
    llvm::outs() << "Parsed MLIR Module:\n";
    module.dump();

    return true;
}

// Bulid TensorGraph from MLIR module
TensorGraph* TosaAnalyzer::buildTensorGraph() {
    TensorGraph* graph = new TensorGraph();
    
    // searching all functions
    module.walk([&](mlir::func::FuncOp funcOp) {
        // find all operation in function
        funcOp.walk([&](mlir::Operation* op) {
            // checking operation
            if (op->getDialect() && op->getDialect()->getNamespace() == "tosa") {
                // create node and add to graph
                TensorNode* node = new TensorNode(op);
                graph->nodes.push_back(node);
                
                // update define node map
                for (auto result : op->getResults()) {
                    graph->definingNodes[result] = node;
                }
                
                // update usage node map
                for (auto operand : op->getOperands()) {
                    graph->userNodes[operand].push_back(node);
                }
            }
        });
    });
    
    return graph;
}

// Export grpah to DOT format for visualization
void TosaAnalyzer::exportGraphToDot(const std::string& filename) {
    std::error_code EC;
    llvm::raw_fd_ostream dotFile(filename, EC);
    
    if (EC) {
        llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
        return;
    }
    
    // DOT file Header
    dotFile << "digraph TensorGraph {\n";
    dotFile << "  node [shape=box];\n";
    
    // define nodes
    for (auto node : graph->nodes) {
        dotFile << "  \"" << node->id << "\" [label=\"" 
                << node->operation->getName().getStringRef().str() << "\"];\n";
    }
    
    // define edges
    for (auto node : graph->nodes) {
        for (auto output : node->outputs) {
            auto usersIt = graph->userNodes.find(output);
            if (usersIt != graph->userNodes.end()) {
                for (auto user : usersIt->second) {
                    dotFile << "  \"" << node->id << "\" -> \"" 
                            << user->id << "\" [label=\"" 
                            << output.getType() << "\"];\n";
                }
            }
        }
    }
    
    dotFile << "}\n";
    dotFile.close();
    
    llvm::outs() << "Graph exported to " << filename << "\n";
}

// perform liveness analysis
void TosaAnalyzer::performLivenessAnalysis() {
    llvm::outs() << "\nPerforming liveness analysis...\n";
    livenessAnalysis = new LivenessAnalysis(graph);
}


// print analysis results
void TosaAnalyzer::printResults() {
    if (livenessAnalysis)
        livenessAnalysis -> printLivenessInfo();
    llvm::outs() << "\nAnalysis complete.\n";
}

// excute tosa analyzer
int TosaAnalyzer::run(int argc, char** argv) {
    // parse command line
    if (!parseCommandLine(argc, argv))
        return 1;

    // load inputfile
    if (!loadAndParseInputFile())
        return 1;

    // build tensor graph
    llvm::outs() << "\nbuild tensor graph\n";
    graph = buildTensorGraph();
    llvm::outs() << "Created" << graph->nodes.size() << " nodes in the dataflow graph\n";

    // Export graph visualization
    exportGraphToDot(outputFileName);

    // Perform liveness analysis
    performLivenessAnalysis();

    // print results
    printResults();

    return 0;
}


int main(int argc, char **argv) {
    TosaAnalyzer analyzer;
    return analyzer.run(argc, argv);
}