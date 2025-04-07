#include "TosaAnalyzer.h"

// MLIR 모듈에서 데이터 흐름 그래프 구성
TensorGraph* buildTensorGraph(mlir::ModuleOp moduleOp) {
    TensorGraph* graph = new TensorGraph();
    
    // 모든 함수를 탐색
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        // 함수 내의 모든 연산 탐색하여 노드 생성
        funcOp.walk([&](mlir::Operation* op) {
            // TOSA 연산인지 확인
            if (op->getDialect() && op->getDialect()->getNamespace() == "tosa") {
                // 노드 생성 및 그래프에 추가
                TensorNode* node = new TensorNode(op);
                graph->nodes.push_back(node);
                
                // 출력값에 대해 정의 노드 맵 업데이트
                for (auto result : op->getResults()) {
                    graph->definingNodes[result] = node;
                }
                
                // 입력값에 대해 사용자 노드 맵 업데이트
                for (auto operand : op->getOperands()) {
                    graph->userNodes[operand].push_back(node);
                }
            }
        });
    });
    
    return graph;
}

// DOT 형식으로 그래프 출력 (Graphviz 시각화용)
void exportGraphToDot(TensorGraph* graph, llvm::StringRef filename) {
    std::error_code EC;
    llvm::raw_fd_ostream dotFile(filename, EC);
    
    if (EC) {
        llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
        return;
    }
    
    // DOT 파일 헤더 작성
    dotFile << "digraph TensorGraph {\n";
    dotFile << "  node [shape=box];\n";
    
    // 노드 정의
    for (auto node : graph->nodes) {
        dotFile << "  \"" << node->id << "\" [label=\"" 
                << node->operation->getName().getStringRef().str() << "\"];\n";
    }
    
    // 에지 정의
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

int main(int argc, char **argv) {
    // LLVM 초기화
    llvm::InitLLVM initLLVM(argc, argv);
    
    // 명령줄 옵션 설정
    llvm::cl::opt<std::string> inputFilename(
        "input",
        llvm::cl::desc("Input MLIR file"),
        llvm::cl::Required);
    
    llvm::cl::opt<std::string> outputDotFile(
        "output-dot",
        llvm::cl::desc("Output DOT file for graph visualization"),
        llvm::cl::init("dataflow.dot"));
    
    llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TOSA Model Liveness Analyzer\n");
    
    // MLIR 컨텍스트 생성
    mlir::MLIRContext context;
    
    // 필요한 다이얼렉트 등록
    context.loadDialect<mlir::tosa::TosaDialect>();
    context.loadDialect<mlir::func::FuncDialect>();
    
    // 소스 파일 로드
    std::string errorMessage;
    auto file = mlir::openInputFile(inputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return 1;
    }
    
    // 소스 관리자 설정
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    
    // MLIR 파싱
    mlir::OwningOpRef<mlir::Operation*> moduleRef = 
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!moduleRef) {
        llvm::errs() << "Error parsing input file\n";
        return 1;
    }
    
    mlir::ModuleOp module = mlir::cast<mlir::ModuleOp>(moduleRef.get());
    
    // MLIR 모듈 내용 출력
    llvm::outs() << "Parsed MLIR Module:\n";
    module.dump();
    
    // 데이터 흐름 그래프 구성
    llvm::outs() << "\nBuilding data flow graph...\n";
    TensorGraph* graph = buildTensorGraph(module);
    llvm::outs() << "Created " << graph->nodes.size() << " nodes in the data flow graph\n";
    
    // 그래프 시각화 (DOT 형식)
    exportGraphToDot(graph, outputDotFile);
    
    // Liveness 분석 수행
    llvm::outs() << "\nPerforming liveness analysis...\n";
    LivenessAnalysis livenessAnalysis(graph);
    
    // 분석 결과 출력
    livenessAnalysis.printLivenessInfo();
    
    // 메모리 정리
    delete graph;
    
    llvm::outs() << "\nAnalysis complete!\n";
    
    return 0;
}