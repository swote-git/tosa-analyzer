#include "LivenessAnalysis.h"

LivenessAnalysis::LivenessAnalysis(TensorGraph *graph) {
    // 1. Topologically sort the graph nodes
    topologicalSort(graph);
    
    // 2. Find defining and using nodes for each value
    buildDefUseInfo(graph);
    
    // 3. liveness analysis in reverse topological order
    computeLiveness();
    
    // 4. Compute live ranges for each value
    computeLiveRanges();
}

// Topologically sort the graph nodes (DFS base)
void LivenessAnalysis::topologicalSort(TensorGraph* graph) {
    std::set<TensorNode*> visited;
    std::set<TensorNode*> temp; // to check cycle
    
    for (auto node : graph->nodes) {
        if (visited.find(node) == visited.end()) {
            topoSortUtil(node, visited, temp, graph);
        }
    }
}

void LivenessAnalysis::topoSortUtil(TensorNode* node, std::set<TensorNode*>& visited, 
                    std::set<TensorNode*>& temp, TensorGraph* graph) {
    // check cycle
    if (temp.find(node) != temp.end()) {
        llvm::errs() << "Error: Cycle detected in graph\n";
        return;
    }
    
    // skip already visited node
    if (visited.find(node) != visited.end()) {
        return;
    }
    
    // check for visit
    temp.insert(node);
    
    // visit all post-order node
    for (auto output : node->outputs) {
        auto users = graph->userNodes.find(output);
        if (users != graph->userNodes.end()) {
            for (auto userNode : users->second) {
                topoSortUtil(userNode, visited, temp, graph);
            }
        }
    }
    
    // check after visit
    temp.erase(node);
    visited.insert(node);
    
    // add node for livenss analysis (in reverse order)
    topoSortedNodes.insert(topoSortedNodes.begin(), node);
}

// 각 값의 정의 노드와 사용 노드 정보 구축
void LivenessAnalysis::buildDefUseInfo(TensorGraph* graph) {
    for (auto node : graph->nodes) {
        // definition node
        for (auto output : node->outputs) {
            liveRanges[output].defNode = node;
        }
        
        // usage node
        for (auto input : node->inputs) {
            if (graph->definingNodes.find(input) != graph->definingNodes.end()) {
                auto defNode = graph->definingNodes[input];
                liveRanges[input].useNodes.push_back(node);
            }
        }
    }
}

// liveness 분석 수행 (위상 정렬의 역순으로 진행)
void LivenessAnalysis::computeLiveness() {
    bool changed = true;
    
    // 모든 노드에 대해 빈 집합으로 초기화
    for (auto node : topoSortedNodes) {
        liveIn[node] = llvm::DenseSet<mlir::Value>();
        liveOut[node] = llvm::DenseSet<mlir::Value>();
    }
    
    // 고정점에 도달할 때까지 반복
    while (changed) {
        changed = false;
        
        // 위상 정렬의 역순으로 모든 노드 처리
        for (auto it = topoSortedNodes.rbegin(); it != topoSortedNodes.rend(); ++it) {
            TensorNode* node = *it;
            
            // 이전 liveIn, liveOut 크기 저장
            size_t oldInSize = liveIn[node].size();
            size_t oldOutSize = liveOut[node].size();
            
            // LIVE_OUT(n) = ∪ LIVE_IN(s)
            for (auto output : node->outputs) {
                for (auto user : liveRanges[output].useNodes) {
                    // 사용자 노드의 liveIn에 있는 모든 값을 현재 노드의 liveOut에 추가
                    for (auto val : liveIn[user]) {
                        liveOut[node].insert(val);
                    }
                }
            }
            
            // LIVE_IN(n) = USE(n) ∪ (LIVE_OUT(n) - DEF(n))
            
            // 먼저 USE(n) 추가
            for (auto input : node->inputs) {
                liveIn[node].insert(input);
            }
            
            // LIVE_OUT(n) - DEF(n) 추가
            for (auto val : liveOut[node]) {
                bool isDefined = false;
                for (auto output : node->outputs) {
                    if (val == output) {
                        isDefined = true;
                        break;
                    }
                }
                
                if (!isDefined) {
                    liveIn[node].insert(val);
                }
            }
            
            // 변화가 있었는지 확인
            if (liveIn[node].size() != oldInSize || liveOut[node].size() != oldOutSize) {
                changed = true;
            }
        }
    }
}

// 각 값의 생존 범위 계산
void LivenessAnalysis::computeLiveRanges() {
    for (auto& entry : liveRanges) {
        mlir::Value val = entry.first;
        LiveRange& range = entry.second;
        
        // 마지막 사용 노드 찾기
        if (!range.useNodes.empty()) {
            // 위상 정렬 순서에 따라 마지막 사용 노드 찾기
            range.lastUseNode = range.useNodes[0];
            int lastUseIdx = -1;
            
            for (auto useNode : range.useNodes) {
                int currentIdx = -1;
                for (size_t i = 0; i < topoSortedNodes.size(); i++) {
                    if (topoSortedNodes[i] == useNode) {
                        currentIdx = i;
                        break;
                    }
                }
                
                if (currentIdx > lastUseIdx) {
                    lastUseIdx = currentIdx;
                    range.lastUseNode = useNode;
                }
            }
        } else {
            // 사용 노드가 없는 경우, 마지막 사용 노드는 없음
            range.lastUseNode = nullptr;
        }
    }
}

// print liveness info
void LivenessAnalysis::printLivenessInfo() {
    llvm::outs() << "Liveness Analysis Results:\n";
    
    // 위상 정렬된 순서대로 노드 정보 출력
    llvm::outs() << "\nExecution Order (Topological Sort):\n";
    for (size_t i = 0; i < topoSortedNodes.size(); i++) {
        llvm::outs() << i << ": " << topoSortedNodes[i]->id << "\n";
    }
    
    // 각 노드의 liveIn, liveOut 정보 출력
    llvm::outs() << "\nNode Liveness Information:\n";
    for (auto node : topoSortedNodes) {
        llvm::outs() << "Node: " << node->id << "\n";
        
        llvm::outs() << "  Live-in: ";
        for (auto val : liveIn[node]) {
            llvm::outs() << val << " ";
        }
        llvm::outs() << "\n";
        
        llvm::outs() << "  Live-out: ";
        for (auto val : liveOut[node]) {
            llvm::outs() << val << " ";
        }
        llvm::outs() << "\n";
        
        llvm::outs() << "  DEF: ";
        for (auto output : node->outputs) {
            llvm::outs() << output << " ";
        }
        llvm::outs() << "\n";
        
        llvm::outs() << "  USE: ";
        for (auto input : node->inputs) {
            llvm::outs() << input << " ";
        }
        llvm::outs() << "\n\n";
    }
    
    // 각 값의 생존 범위 정보 출력
    llvm::outs() << "\nTensor Value Live Ranges:\n";
    for (auto& entry : liveRanges) {
        mlir::Value val = entry.first;
        LiveRange& range = entry.second;
        
        llvm::outs() << "Value: " << val << " (Type: " << val.getType() << ")\n";
        llvm::outs() << "  Defined at: " << range.defNode->id << "\n";
        
        llvm::outs() << "  Used at: ";
        for (auto useNode : range.useNodes) {
            llvm::outs() << useNode->id << " ";
        }
        llvm::outs() << "\n";
        
        if (range.lastUseNode) {
            llvm::outs() << "  Last use at: " << range.lastUseNode->id << "\n";
        } else {
            llvm::outs() << "  No uses (dead code)\n";
        }
        
        llvm::outs() << "\n";
    }
    
    // 메모리 최적화 기회 출력
    llvm::outs() << "\nMemory Optimization Opportunities:\n";
    for (size_t i = 0; i < topoSortedNodes.size(); i++) {
        TensorNode* node = topoSortedNodes[i];
        
        llvm::outs() << "After node " << node->id << ":\n";
        
        // 이 노드에서 죽는 값들 찾기
        std::vector<mlir::Value> dyingValues;
        for (auto& entry : liveRanges) {
            mlir::Value val = entry.first;
            LiveRange& range = entry.second;
            
            if (range.lastUseNode == node) {
                dyingValues.push_back(val);
            }
        }
        
        if (!dyingValues.empty()) {
            llvm::outs() << "  Can free memory for: ";
            for (auto val : dyingValues) {
                llvm::outs() << val << " ";
            }
            llvm::outs() << "\n";
        } else {
            llvm::outs() << "  No memory can be freed\n";
        }
    }
}
