#include "TensorGraph.h"

// 소멸자: 노드 메모리 정리
TensorGraph::~TensorGraph() {
    for (auto node : nodes) {
        delete node;
    }
}
