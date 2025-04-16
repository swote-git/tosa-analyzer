#include "TosaAnalyzer.h"
#include "LivenessAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "TensorNode.h"

// constructor
TosaAnalyzer::TosaAnalyzer() : graph(nullptr), liveness(nullptr), memoryPlanner(nullptr) {
    // register dialect
    context.loadDialect<mlir::tosa::TosaDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::quant::QuantDialect>();
}

// Destructor
TosaAnalyzer::~TosaAnalyzer() {
    if (memoryPlanner)
        delete memoryPlanner;
    if (liveness) 
        delete liveness;
    if (graph)
        delete graph;
}

// parse input from command line
bool TosaAnalyzer::parseCommandLine(int argc, char **argv) {
    // LLVM initialization
    llvm::InitLLVM initLLVM(argc, argv);
    
    // command line options
    llvm::cl::opt<std::string> inputFile(
        "input",
        llvm::cl::desc("Input MLIR file"),
        llvm::cl::Required);
    
    llvm::cl::opt<std::string> livenessVisualOutput(
        "liveness-visulize-dot",
        llvm::cl::desc("Output DOT file for graph visualization"),
        llvm::cl::init("tensorGrpah.dot"));

    llvm::cl::opt<std::string> memoryPlanOutput(
        "memory-plan",
        llvm::cl::desc("Output file for memory allocation plan"),
        llvm::cl::init("memory_plan.h"));
        
    llvm::cl::opt<std::string> memoryVisualOutput(
        "memory-vis",
        llvm::cl::desc("Output file for memory usage visualization"),
        llvm::cl::init("memory_vis.html"));
    
    llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TOSA Model Liveness Analyzer\n");

    // store file values
    inputFileName = inputFile;
    livenessVisualFile = livenessVisualOutput;
    memoryPlanFile = memoryPlanOutput;
    memoryVisualFile = memoryVisualOutput;

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
    liveness = new LivenessAnalysis(graph);
}

void TosaAnalyzer::planMemoryAllocation() {
    if (!liveness) {
        llvm::errs() << "Error: Liveness analysis must be performed before memory planning\n";
        return;
    }
    
    llvm::outs() << "\nPlanning memory allocation...\n";
    
    // Create memory planner
    memoryPlanner = new MemoryPlanner(graph, liveness);
    
    // Compute tensor sizes
    memoryPlanner->computeTensorSizes();
    
    // Build allocation plan
    memoryPlanner->buildAllocationPlan();
    
    // Optimize memory reuse
    memoryPlanner->performMemoryOptimizer();
    
    // Generate allocation code
    memoryPlanner->generateAllocationCode(memoryPlanFile);
    
    // Generate memory visualization
    memoryPlanner->visualizeMemoryUsage(memoryVisualFile);
    
    llvm::outs() << "Memory planning complete!\n";
}

// print analysis results
void TosaAnalyzer::printResults() {
    if (liveness)
        liveness->printLivenessInfo();

    if (memoryPlanner)
        memoryPlanner->printMemoryStatistics();

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
    exportGraphToDot(livenessVisualFile);

    // Perform liveness analysis
    performLivenessAnalysis();

    // Plan memory allocation
    planMemoryAllocation();
    
    // print results
    printResults();

    return 0;
}


int main(int argc, char **argv) {
    TosaAnalyzer analyzer;
    return analyzer.run(argc, argv);
}