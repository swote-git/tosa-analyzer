# TOSA Analyzer

TOSA Analyzer is a static analysis tool for MLIR models with TOSA operations. It provides liveness analysis and memory optimization capabilities to help developers understand and optimize memory usage patterns in tensor computation graphs.

## Features

- **Data Flow Graph Construction**: Builds a comprehensive graph representation of TOSA operations and their data dependencies.
- **Liveness Analysis**: Tracks when tensors are created and when they're last used to identify memory reuse opportunities.
- **Memory Planning**: Implements various memory allocation strategies (including Heavy-Light Decomposition) for optimizing memory usage.
- **Visualization**: Generates DOT files for visualizing the data flow graph and memory usage patterns.
- **Memory Usage Metrics**: Provides detailed metrics about memory efficiency, including peak usage and potential optimizations.

## Installation

### Prerequisites

- CMake (3.13.4 or higher)
- LLVM/MLIR development libraries (latest version recommended)
- Clang compiler
- Ninja build system (recommended)

### Building from Source

1. First, ensure you have LLVM and MLIR built with the required dialects:

```bash
# Clone LLVM project if you don't have it already
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create a build directory
mkdir build && cd build

# Configure with all necessary dialects
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_ENABLE_RTTI=ON \
  -DMLIR_ENABLE_DIALECT_AFFINE=ON \
  -DMLIR_ENABLE_DIALECT_ARITH=ON \
  -DMLIR_ENABLE_DIALECT_FUNC=ON \
  -DMLIR_ENABLE_DIALECT_TENSOR=ON \
  -DMLIR_ENABLE_DIALECT_TOSA=ON \
  -DMLIR_ENABLE_DIALECT_QUANT=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build LLVM/MLIR (this may take a while)
ninja
```

2. Clone and build TOSA Analyzer:


```bash
# Clone the repository
git clone https://github.com/yourusername/tosa-analyzer.git
cd tosa-analyzer

# Create and navigate to build directory
mkdir build && cd build

# Configure 
cmake -G Ninja ..

# Build
ninja
```

## Usage

### Basic Usage

```bash
# Run liveness analysis on a TOSA model
./TosaAnalyzer --input path/to/your/model.mlir --output-dot graph.dot
```

### Command Line Options

- --input <file>: Input MLIR file (required)
- --output-dot <file>: Output DOT file for graph visualization (default: dataflow.dot)
- --memory-code <file>: Output file for memory allocation code (default: memory_plan.h)
- --memory-vis <file>: Output file for memory usage visualization (default: memory_vis.html)

### Visualization 

The tool generates two types of visualizations:

1. Data Flow Graph: A DOT file visualization of the tensor operations and their dependencies, which can be viewed with tools like Graphviz.
2. Memory Usage Timeline: An HTML visualization showing when tensors are allocated and freed during execution.

### Memory Optimization

TOSA Analyzer implements several memory optimization strategies:

- First-Fit: A simple greedy algorithm that allocates tensors in the first available space.
- Heavy-Light Decomposition: A sophisticated algorithm that separates tensors into "heavy" (large) and "light" (small) categories and optimizes them separately.

The optimization results include:

- Total memory requirements
- Peak memory usage
- Memory efficiency metrics
- Suggested allocation plan