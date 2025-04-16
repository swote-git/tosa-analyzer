#include "MemoryPlanner.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

MemoryPlanner::MemoryPlanner(TensorGraph* graph, LivenessAnalysis* liveness)
    : graph(graph), liveness(liveness), totalMemory(0), peakMemory(0) {
}

MemoryPlanner::~MemoryPlanner() {
    // No manual cleanup needed - using STL containers
}

void MemoryPlanner::computeTensorSizes() {
    llvm::outs() << "Computing tensor sizes...\n";
    
    // Process all values in the liveness analysis
    for (auto& entry : liveness->liveRanges) {
        mlir::Value val = entry.getFirst();
        auto& range = entry.getSecond();
        
        // Skip non-tensor values
        if (!mlir::isa<mlir::ShapedType>(val.getType()))
            continue;
            
        // Create allocation info
        AllocationInfo alloc;
        alloc.value = val;
        alloc.defNode = range.defNode;
        alloc.lastUseNode = range.lastUseNode;
        alloc.size = estimateTensorSize(val);
        alloc.allocatedPoolIndex = -1;
        alloc.offset = -1;
        
        // Check if this is an input or output
        alloc.isModelInput = (range.defNode == nullptr); // No defining node means it's an input
        alloc.isModelOutput = true; // Assume it's an output by default
        
        // If this value is used by any node, it's not an output
        if (!range.useNodes.empty()) {
            alloc.isModelOutput = false;
        }
        
        // Special case: if it's the result of the last node and has no uses, it's an output
        if (range.defNode && range.useNodes.empty()) {
            alloc.isModelOutput = true;
        }
        
        // Add to our allocation list
        allocations.push_back(alloc);
        
        // Update total memory
        totalMemory += alloc.size;
    }
    
    llvm::outs() << "Found " << allocations.size() << " tensors requiring allocation\n";
    llvm::outs() << "Total memory for all tensors: " << (totalMemory / 1024.0) << " KB\n";
}

size_t MemoryPlanner::estimateTensorSize(mlir::Value tensor) {
    // Get the tensor type
    auto type =  mlir::dyn_cast<mlir::ShapedType>(tensor.getType());
    if (!type) {
        return 0; // Not a tensor
    }
    
    // Calculate number of elements
    int64_t numElements = 1;
    for (auto dim : type.getShape()) {
        if (dim < 0) {
            // Dynamic dimension, use a default size
            dim = 1;
        }
        numElements *= dim;
    }
    
    // Determine element size based on element type
    size_t elementSize = 4; // Default to float32 (4 bytes)
    
    if (type.getElementType().isIntOrIndex()) {
        // Integer type
        unsigned bitWidth = 32; // Default to 32-bit
        
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type.getElementType())) {
            bitWidth = intType.getWidth();
        }
        
        elementSize = (bitWidth + 7) / 8; // Round up to nearest byte
    }
    else if (type.getElementType().isF16()) {
        elementSize = 2; // 16-bit float
    }
    else if (type.getElementType().isF64()) {
        elementSize = 8; // 64-bit float
    }
    
    return numElements * elementSize;
}

void MemoryPlanner::buildAllocationPlan() {
    llvm::outs() << "Building memory allocation plan...\n";
    
    // Sort tensors by definition time (based on the topological order)
    std::vector<int> topoOrder(liveness->topoSortedNodes.size());
    for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
        topoOrder[i] = i;
    }
    
    // Sort allocations by definition time
    std::sort(allocations.begin(), allocations.end(), 
        [&](const AllocationInfo& a, const AllocationInfo& b) {
            // If one is a model input, it comes first
            if (a.isModelInput && !b.isModelInput) return true;
            if (!a.isModelInput && b.isModelInput) return false;
            
            // Otherwise, sort by definition node in topological order
            if (a.defNode && b.defNode) {
                int aIdx = -1, bIdx = -1;
                for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                    if (liveness->topoSortedNodes[i] == a.defNode) aIdx = i;
                    if (liveness->topoSortedNodes[i] == b.defNode) bIdx = i;
                }
                return aIdx < bIdx;
            }
            
            return false;
    });
    
    // Create initial memory pool
    createNewMemoryPool(1024 * 1024); // 1MB initial pool
    
    // Allocate each tensor
    for (auto& alloc : allocations) {
        if (alloc.isModelInput || alloc.isModelOutput) {
            // Model inputs and outputs go to a special pool (0)
            alloc.allocatedPoolIndex = 0;
            alloc.offset = 0; // For simplicity, we're placing them at offset 0
        } else {
            // Find a suitable memory pool for this tensor
            int poolIndex = findMemoryPool(alloc);
            if (poolIndex < 0) {
                // No suitable pool found, create a new one
                poolIndex = createNewMemoryPool(std::max(alloc.size, (size_t)1024 * 1024));
            }
            
            // Insert into the pool
            insertIntoMemoryPool(poolIndex, alloc);
        }
    }
    
    // Build memory usage timeline
    buildMemoryUsageTimeline();
    
    // Calculate peak memory usage
    peakMemory = *std::max_element(memoryUsageTimeline.begin(), memoryUsageTimeline.end());
    
    llvm::outs() << "Memory allocation plan complete.\n";
    llvm::outs() << "Total memory: " << (totalMemory / 1024.0) << " KB\n";
    llvm::outs() << "Peak memory: " << (peakMemory / 1024.0) << " KB\n";
    llvm::outs() << "Memory efficiency: " << ((float)totalMemory / peakMemory) * 100.0f << "%\n";
}

int MemoryPlanner::findMemoryPool(AllocationInfo& alloc) {
    // Try to find a memory pool that has enough free space
    for (size_t i = 0; i < memoryPools.size(); i++) {
        auto& pool = memoryPools[i];
        
        // Check each free interval
        for (auto& interval : pool.freeIntervals) {
            size_t intervalSize = interval.second - interval.first;
            if (intervalSize >= alloc.size) {
                return i; // Found a suitable pool
            }
        }
    }
    
    return -1; // No suitable pool found
}

void MemoryPlanner::insertIntoMemoryPool(int poolIndex, AllocationInfo& alloc) {
    auto& pool = memoryPools[poolIndex];
    
    // Find a free interval that's large enough
    for (auto it = pool.freeIntervals.begin(); it != pool.freeIntervals.end(); ++it) {
        size_t intervalSize = it->second - it->first;
        if (intervalSize >= alloc.size) {
            // Found a suitable interval
            alloc.allocatedPoolIndex = poolIndex;
            alloc.offset = it->first;
            
            // Update the free interval
            int newStart = it->first + alloc.size;
            int oldEnd = it->second;
            
            // Remove this interval
            it = pool.freeIntervals.erase(it);
            
            // If there's still free space after this allocation, add a new interval
            if (newStart < oldEnd) {
                pool.freeIntervals.push_back(std::make_pair(newStart, oldEnd));
            }
            
            // Sort free intervals by start offset
            std::sort(pool.freeIntervals.begin(), pool.freeIntervals.end(),
                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                    return a.first < b.first;
                });
            
            return;
        }
    }
    
    // This should not happen if findMemoryPool returned a valid pool
    llvm::errs() << "Error: Failed to insert tensor into memory pool\n";
}

int MemoryPlanner::createNewMemoryPool(size_t initialSize) {
    MemoryPool pool;
    pool.size = initialSize;
    pool.name = "pool_" + std::to_string(memoryPools.size());
    
    // Initially, the entire pool is free
    pool.freeIntervals.push_back(std::make_pair(0, initialSize));
    
    memoryPools.push_back(pool);
    return memoryPools.size() - 1;
}

void MemoryPlanner::buildMemoryUsageTimeline() {
    // Initialize timeline with size equal to the number of nodes
    memoryUsageTimeline.resize(liveness->topoSortedNodes.size() + 1, 0);
    
    // For each tensor, add its size to the timeline from def to last use
    for (const auto& alloc : allocations) {
        // Find indices of def and last use nodes
        int defIdx = 0;
        int lastUseIdx = liveness->topoSortedNodes.size();
        
        if (alloc.defNode) {
            for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                if (liveness->topoSortedNodes[i] == alloc.defNode) {
                    defIdx = i;
                    break;
                }
            }
        }
        
        if (alloc.lastUseNode) {
            for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                if (liveness->topoSortedNodes[i] == alloc.lastUseNode) {
                    lastUseIdx = i;
                    break;
                }
            }
        }
        
        // Add this tensor's size to the timeline
        for (int i = defIdx; i <= lastUseIdx; i++) {
            memoryUsageTimeline[i] += alloc.size;
        }
    }
}

void MemoryPlanner::performMemoryOptimizer() {
    llvm::outs() << "\nOptimizing memory reuse using heavy-light decomposition...\n";
    
    // Reset memory pools for a fresh optimization
    memoryPools.clear();
    
    // Create initial pool for inputs and outputs
    createNewMemoryPool(1024 * 1024); // 1MB initial pool
    
    // Step 1: Classify tensors as "heavy" or "light"
    const float HEAVY_THRESHOLD = 0.1; // Tensors using >10% of total memory are "heavy"
    size_t sizeThreshold = totalMemory * HEAVY_THRESHOLD;
    
    std::vector<AllocationInfo*> heavyTensors;
    std::vector<AllocationInfo*> lightTensors;
    
    for (auto& alloc : allocations) {
        // Model inputs and outputs should be handled separately
        if (alloc.isModelInput || alloc.isModelOutput) {
            // Fixed allocation in pool 0
            alloc.allocatedPoolIndex = 0;
            alloc.offset = 0; // For simplicity, we're placing them at offset 0
            continue;
        }
        
        // Reset allocation info for re-optimization
        alloc.allocatedPoolIndex = -1;
        alloc.offset = -1;
        
        // Classify as heavy or light
        if (alloc.size >= sizeThreshold) {
            heavyTensors.push_back(&alloc);
        } else {
            lightTensors.push_back(&alloc);
        }
    }
    
    llvm::outs() << "  Classification: " << heavyTensors.size() << " heavy tensors, " 
                 << lightTensors.size() << " light tensors\n";
    
    // Step 2: Sort heavy tensors by size (largest first)
    std::sort(heavyTensors.begin(), heavyTensors.end(), 
        [](const AllocationInfo* a, const AllocationInfo* b) {
            return a->size > b->size;
        });
    
    // Step 3: Sort light tensors by lifetime (earliest definition first)
    std::sort(lightTensors.begin(), lightTensors.end(), 
        [this](const AllocationInfo* a, const AllocationInfo* b) {
            if (!a->defNode || !b->defNode) {
                return a->defNode == nullptr; // Null nodes come first
            }
            
            int aIdx = -1, bIdx = -1;
            for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
                if (liveness->topoSortedNodes[i] == a->defNode) aIdx = i;
                if (liveness->topoSortedNodes[i] == b->defNode) bIdx = i;
            }
            return aIdx < bIdx;
        });
    
    // Step 4: Allocate heavy tensors first (each in its own dedicated pool)
    for (auto heavyAlloc : heavyTensors) {
        int poolIdx = createNewMemoryPool(heavyAlloc->size + 1024); // Add a small buffer
        heavyAlloc->allocatedPoolIndex = poolIdx;
        heavyAlloc->offset = 0;
        
        // Mark the space as used
        auto& pool = memoryPools[poolIdx];
        pool.freeIntervals.clear(); // No free space in this dedicated pool
        
        llvm::outs() << "  Allocated heavy tensor (" << (heavyAlloc->size / 1024.0) 
                     << " KB) to dedicated pool " << poolIdx << "\n";
    }
    
    // Step 5: Create a shared pool for light tensors
    int lightPoolIdx = createNewMemoryPool(totalMemory / 2); // Start with half of total memory
    
    // Step 6: Allocate light tensors using the best-fit algorithm
    for (auto lightAlloc : lightTensors) {
        bool allocated = false;
        
        // First, try to find a spot in an existing pool 
        // (both fit size and no lifetime conflict)
        for (size_t poolIdx = 1; poolIdx < memoryPools.size(); poolIdx++) {
            auto& pool = memoryPools[poolIdx];
            
            // Skip dedicated heavy tensor pools (they have no free intervals)
            if (pool.freeIntervals.empty()) continue;
            
            // Check each free interval in this pool
            for (auto intervalIt = pool.freeIntervals.begin(); 
                 intervalIt != pool.freeIntervals.end(); ++intervalIt) {
                
                size_t intervalSize = intervalIt->second - intervalIt->first;
                
                // If interval is large enough
                if (intervalSize >= lightAlloc->size) {
                    // Check for lifetime conflicts with all tensors in this pool
                    bool hasConflict = false;
                    
                    // Get definition and last use indices for this tensor
                    int defIdx = getNodeIndex(lightAlloc->defNode);
                    int lastUseIdx = getNodeIndex(lightAlloc->lastUseNode);
                    
                    // Check against all allocated tensors in this pool
                    for (const auto& otherAlloc : allocations) {
                        if (otherAlloc.allocatedPoolIndex != poolIdx || 
                            &otherAlloc == lightAlloc) {
                            continue;
                        }
                        
                        int otherDefIdx = getNodeIndex(otherAlloc.defNode);
                        int otherLastUseIdx = getNodeIndex(otherAlloc.lastUseNode);
                        
                        // Check for overlap in lifetimes
                        if (!(lastUseIdx < otherDefIdx || otherLastUseIdx < defIdx)) {
                            hasConflict = true;
                            break;
                        }
                    }
                    
                    // If no conflicts, allocate here
                    if (!hasConflict) {
                        lightAlloc->allocatedPoolIndex = poolIdx;
                        lightAlloc->offset = intervalIt->first;
                        
                        // Update free intervals
                        int newStart = intervalIt->first + lightAlloc->size;
                        int oldEnd = intervalIt->second;
                        
                        // Remove this interval
                        intervalIt = pool.freeIntervals.erase(intervalIt);
                        
                        // If there's still free space after this allocation, add a new interval
                        if (newStart < oldEnd) {
                            pool.freeIntervals.push_back(std::make_pair(newStart, oldEnd));
                            
                            // Sort free intervals by start offset
                            std::sort(pool.freeIntervals.begin(), pool.freeIntervals.end(),
                                [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                    return a.first < b.first;
                                });
                        }
                        
                        allocated = true;
                        break;
                    }
                }
            }
            
            if (allocated) break;
        }
        
        // If we couldn't fit it in any existing pool, use the light tensor pool
        if (!allocated) {
            auto& lightPool = memoryPools[lightPoolIdx];
            
            // Find a free interval that's large enough
            bool foundInterval = false;
            for (auto intervalIt = lightPool.freeIntervals.begin(); 
                 intervalIt != lightPool.freeIntervals.end(); ++intervalIt) {
                
                size_t intervalSize = intervalIt->second - intervalIt->first;
                
                if (intervalSize >= lightAlloc->size) {
                    lightAlloc->allocatedPoolIndex = lightPoolIdx;
                    lightAlloc->offset = intervalIt->first;
                    
                    // Update free intervals
                    int newStart = intervalIt->first + lightAlloc->size;
                    int oldEnd = intervalIt->second;
                    
                    // Remove this interval
                    intervalIt = lightPool.freeIntervals.erase(intervalIt);
                    
                    // If there's still free space after this allocation, add a new interval
                    if (newStart < oldEnd) {
                        lightPool.freeIntervals.push_back(std::make_pair(newStart, oldEnd));
                        
                        // Sort free intervals by start offset
                        std::sort(lightPool.freeIntervals.begin(), lightPool.freeIntervals.end(),
                            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                return a.first < b.first;
                            });
                    }
                    
                    foundInterval = true;
                    allocated = true;
                    break;
                }
            }
            
            // If no suitable interval found, expand the light pool
            if (!foundInterval) {
                // Calculate expansion size (at least tensor size, but add some buffer)
                size_t expansionSize = std::max(lightAlloc->size * 2, (size_t)1024 * 64);
                
                // Get current pool size
                size_t oldSize = lightPool.size;
                
                // Expand the pool
                lightPool.size += expansionSize;
                
                // Add the expansion as a new free interval
                lightPool.freeIntervals.push_back(std::make_pair(oldSize, lightPool.size));
                
                // Now allocate the tensor at the start of the new interval
                lightAlloc->allocatedPoolIndex = lightPoolIdx;
                lightAlloc->offset = oldSize;
                
                // Update the new interval
                int newStart = oldSize + lightAlloc->size;
                
                // Remove last interval we just added
                lightPool.freeIntervals.pop_back();
                
                // Add the remaining free space if any
                if (newStart < lightPool.size) {
                    lightPool.freeIntervals.push_back(std::make_pair(newStart, lightPool.size));
                }
                
                allocated = true;
                
                llvm::outs() << "  Expanded light tensor pool by " << (expansionSize / 1024.0) 
                             << " KB to fit tensor of size " << (lightAlloc->size / 1024.0) 
                             << " KB\n";
            }
        }
        
        if (!allocated) {
            llvm::errs() << "  Error: Failed to allocate tensor of size " 
                         << (lightAlloc->size / 1024.0) << " KB\n";
        }
    }
    
    // Step 7: Merge adjacent free intervals in all pools
    compactMemoryPools();
    
    // Step 8: Build memory usage timeline
    buildMemoryUsageTimeline();
    
    // Calculate peak memory usage
    peakMemory = *std::max_element(memoryUsageTimeline.begin(), memoryUsageTimeline.end());
    
    // Memory efficiency metrics
    float naiveEfficiency = ((float)totalMemory / peakMemory) * 100.0f;
    
    // Calculate total pool size
    size_t totalPoolSize = 0;
    for (const auto& pool : memoryPools) {
        totalPoolSize += pool.size;
    }
    
    float poolEfficiency = ((float)totalMemory / totalPoolSize) * 100.0f;
    
    llvm::outs() << "Memory optimization complete.\n";
    llvm::outs() << "  Total memory: " << (totalMemory / 1024.0) << " KB\n";
    llvm::outs() << "  Peak memory usage: " << (peakMemory / 1024.0) << " KB\n";
    llvm::outs() << "  Total pool size: " << (totalPoolSize / 1024.0) << " KB\n";
    llvm::outs() << "  Temporal efficiency: " << naiveEfficiency << "%\n";
    llvm::outs() << "  Spatial efficiency: " << poolEfficiency << "%\n";
}

// Helper method to get node index in topological order
int MemoryPlanner::getNodeIndex(TensorNode* node) {
    if (!node) return liveness->topoSortedNodes.size(); // Default to end of execution
    
    for (size_t i = 0; i < liveness->topoSortedNodes.size(); i++) {
        if (liveness->topoSortedNodes[i] == node) return i;
    }
    
    return -1; // Not found (shouldn't happen)
}

void MemoryPlanner::compactMemoryPools() {
    for (auto& pool : memoryPools) {
        if (pool.freeIntervals.empty()) continue;
        
        // Sort by start offset
        std::sort(pool.freeIntervals.begin(), pool.freeIntervals.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.first < b.first;
            });
        
        // Merge adjacent intervals
        std::vector<std::pair<int, int>> mergedIntervals;
        mergedIntervals.push_back(pool.freeIntervals[0]);
        
        for (size_t i = 1; i < pool.freeIntervals.size(); i++) {
            auto& last = mergedIntervals.back();
            auto& current = pool.freeIntervals[i];
            
            if (last.second == current.first) {
                // Adjacent intervals, merge them
                last.second = current.second;
            } else {
                // Non-adjacent, add as a new interval
                mergedIntervals.push_back(current);
            }
        }
        
        pool.freeIntervals = mergedIntervals;
    }
}

void MemoryPlanner::generateAllocationCode(const std::string& filename) {
    std::error_code EC;
    llvm::raw_fd_ostream outFile(filename, EC);
    
    if (EC) {
        llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
        return;
    }
    
    outFile << "// Memory allocation plan generated by TosaAnalyzer\n\n";
    
    // Generate memory pool declarations
    outFile << "// Memory pool declarations\n";
    for (size_t i = 0; i < memoryPools.size(); i++) {
        outFile << "uint8_t " << memoryPools[i].name << "[" << memoryPools[i].size << "];\n";
    }
    outFile << "\n";
    
    // Generate tensor allocation macros
    outFile << "// Tensor allocation macros\n";
    for (size_t i = 0; i < allocations.size(); i++) {
        const auto& alloc = allocations[i];
        
        // Generate tensor name based on definition node or index
        std::string tensorName;
        if (alloc.defNode) {
            tensorName = "tensor_" + alloc.defNode->id;
        } else {
            tensorName = "input_tensor_" + std::to_string(i);
        }
        
        // Clean up name for C macro (remove invalid characters)
        std::replace(tensorName.begin(), tensorName.end(), '.', '_');
        std::replace(tensorName.begin(), tensorName.end(), '-', '_');
        
        outFile << "#define " << tensorName << " ((";
        
        // Try to determine the appropriate C type
        auto type = mlir::dyn_cast<mlir::ShapedType>(alloc.value.getType());
        if (type) {
            if (type.getElementType().isF32()) {
                outFile << "float*";
            } else if (type.getElementType().isF64()) {
                outFile << "double*";
            } else if (type.getElementType().isInteger(8)) {
                outFile << "int8_t*";
            } else if (type.getElementType().isInteger(16)) {
                outFile << "int16_t*";
            } else if (type.getElementType().isInteger(32)) {
                outFile << "int32_t*";
            } else if (type.getElementType().isInteger(64)) {
                outFile << "int64_t*";
            } else {
                outFile << "void*";
            }
        } else {
            outFile << "void*";
        }
        
        outFile << ")(";
        
        if (alloc.allocatedPoolIndex >= 0 && alloc.allocatedPoolIndex < memoryPools.size()) {
            outFile << memoryPools[alloc.allocatedPoolIndex].name;
            if (alloc.offset > 0) {
                outFile << " + " << alloc.offset;
            }
        } else {
            outFile << "NULL /* Error: No memory pool assigned */";
        }
        
        outFile << "))\n";
    }
    
    outFile << "\n// End of memory allocation plan\n";
    outFile.close();
    
    llvm::outs() << "Generated memory allocation code to " << filename << "\n";
}

void MemoryPlanner::printMemoryStatistics() {
    llvm::outs() << "\n===== Memory Usage Statistics =====\n";
    llvm::outs() << "Total number of tensors: " << allocations.size() << "\n";
    llvm::outs() << "Total memory for all tensors: " << (totalMemory / 1024.0) << " KB\n";
    llvm::outs() << "Peak memory usage: " << (peakMemory / 1024.0) << " KB\n";
    llvm::outs() << "Memory efficiency: " << ((float)totalMemory / peakMemory) * 100.0f << "%\n";
    
    // Print memory pool information
    llvm::outs() << "\nMemory Pools:\n";
    for (size_t i = 0; i < memoryPools.size(); i++) {
        auto& pool = memoryPools[i];
        
        // Calculate used space
        size_t freeSpace = 0;
        for (auto& interval : pool.freeIntervals) {
            freeSpace += (interval.second - interval.first);
        }
        size_t usedSpace = pool.size - freeSpace;
        
        llvm::outs() << "  Pool " << i << " (" << pool.name << "): " 
                     << (pool.size / 1024.0) << " KB total, " 
                     << (usedSpace / 1024.0) << " KB used (" 
                     << (usedSpace * 100.0 / pool.size) << "%)\n";
    }
    
    // Print largest tensors
    llvm::outs() << "\nLargest Tensors:\n";
    
    // Copy and sort allocations by size
    std::vector<AllocationInfo> sortedAllocs = allocations;
    std::sort(sortedAllocs.begin(), sortedAllocs.end(), 
        [](const AllocationInfo& a, const AllocationInfo& b) {
            return a.size > b.size;
        });
    
    // Print top 10 or fewer
    size_t numToPrint = std::min(sortedAllocs.size(), (size_t)10);
    for (size_t i = 0; i < numToPrint; i++) {
        auto& alloc = sortedAllocs[i];
        std::string nodeName = alloc.defNode ? alloc.defNode->id : "MODEL_INPUT";
        
        llvm::outs() << "  " << i+1 << ". " << nodeName << ": " 
                     << (alloc.size / 1024.0) << " KB";
        
        // Show tensor shape if available
        auto type =  mlir::dyn_cast<mlir::ShapedType>(alloc.value.getType());
        if (type) {
            llvm::outs() << " (shape: [";
            auto shape = type.getShape();
            for (size_t j = 0; j < shape.size(); j++) {
                if (j > 0) llvm::outs() << ", ";
                llvm::outs() << shape[j];
            }
            llvm::outs() << "])";
        }
        
        llvm::outs() << "\n";
    }
    
    // Print memory usage timeline
    llvm::outs() << "\nMemory Usage Timeline:\n";
    
    // Find max for scaling
    size_t maxMemory = *std::max_element(memoryUsageTimeline.begin(), memoryUsageTimeline.end());
    const int chartWidth = 50;
    
    for (size_t i = 0; i < memoryUsageTimeline.size(); i++) {
        std::string nodeName = (i < liveness->topoSortedNodes.size()) ? 
                               liveness->topoSortedNodes[i]->id : "END";
        
        // Scale to chart width
        int barLength = (memoryUsageTimeline[i] * chartWidth) / (maxMemory > 0 ? maxMemory : 1);
        
        llvm::outs() << "  " << i << ": " << nodeName << " - " 
                     << (memoryUsageTimeline[i] / 1024.0) << " KB ";
        
        // Print bar
        llvm::outs() << "[";
        for (int j = 0; j < barLength; j++) {
            llvm::outs() << "=";
        }
        for (int j = barLength; j < chartWidth; j++) {
            llvm::outs() << " ";
        }
        llvm::outs() << "]\n";
    }
}

void MemoryPlanner::visualizeMemoryUsage(const std::string& filename) {
    std::error_code EC;
    llvm::raw_fd_ostream outFile(filename, EC);
    
    if (EC) {
        llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
        return;
    }
    
    // Generate a HTML file with a memory visualization
    outFile << "<!DOCTYPE html>\n"
            << "<html>\n"
            << "<head>\n"
            << "  <title>Memory Usage Visualization</title>\n"
            << "  <style>\n"
            << "    body { font-family: Arial, sans-serif; margin: 20px; }\n"
            << "    .tensor { position: absolute; border: 1px solid #000; background: #ACE; }\n"
            << "    .tensor-label { font-size: 10px; overflow: hidden; }\n"
            << "    .pool { position: relative; border: 1px solid #444; margin-bottom: 10px; }\n"
            << "    .pool-label { font-weight: bold; margin-bottom: 5px; }\n"
            << "    .timeline { margin-top: 30px; }\n"
            << "    .timeline-bar { height: 20px; background: #ACE; margin-bottom: 2px; }\n"
            << "  </style>\n"
            << "</head>\n"
            << "<body>\n"
            << "  <h1>Memory Usage Visualization</h1>\n"
            << "  <h2>Statistics</h2>\n"
            << "  <p>Total Tensors: " << allocations.size() << "</p>\n"
            << "  <p>Total Memory: " << (totalMemory / 1024.0) << " KB</p>\n"
            << "  <p>Peak Memory: " << (peakMemory / 1024.0) << " KB</p>\n"
            << "  <p>Memory Efficiency: " << ((float)totalMemory / peakMemory) * 100.0f << "%</p>\n"
            << "\n  <h2>Memory Pools</h2>\n";
    
    // Visualize each memory pool
    const int poolWidth = 800;
    const int poolHeight = 100;
    
    for (size_t i = 0; i < memoryPools.size(); i++) {
        auto& pool = memoryPools[i];
        
        outFile << "  <div class='pool-label'>Pool " << i << " (" << pool.name 
                << "): " << (pool.size / 1024.0) << " KB</div>\n"
                << "  <div class='pool' style='width: " << poolWidth 
                << "px; height: " << poolHeight << "px;'>\n";
        
        // Draw each tensor in this pool
        for (const auto& alloc : allocations) {
            if (alloc.allocatedPoolIndex != i) continue;
            
            // Calculate position and size
            double xRatio = (double)alloc.offset / pool.size;
            double widthRatio = (double)alloc.size / pool.size;
            
            int x = xRatio * poolWidth;
            int width = std::max(1, (int)(widthRatio * poolWidth));
            
            std::string nodeName = alloc.defNode ? alloc.defNode->id : "MODEL_INPUT";
            
            // Generate a color based on the tensor type
            std::string color = "#ACE"; // Default blue
            auto type =  mlir::dyn_cast<mlir::ShapedType>(alloc.value.getType());
            if (type) {
                if (type.getElementType().isF32()) color = "#AEC";
                else if (type.getElementType().isInteger(8)) color = "#ECA";
                else if (type.getElementType().isInteger(16)) color = "#CDE";
                else if (type.getElementType().isInteger(32)) color = "#EAC";
            }
            
            outFile << "    <div class='tensor' style='left: " << x << "px; top: 10px; width: "
                    << width << "px; height: " << (poolHeight - 20) << "px; background: " << color << ";'>\n"
                    << "      <div class='tensor-label'>" << nodeName << "<br>"
                    << (alloc.size / 1024.0) << " KB</div>\n"
                    << "    </div>\n";
        }
        
        outFile << "  </div>\n\n";
    }
    
    // Draw the memory usage timeline
    outFile << "  <h2>Memory Usage Timeline</h2>\n"
            << "  <div class='timeline'>\n";
    
    // Find max for scaling
    size_t maxMemory = *std::max_element(memoryUsageTimeline.begin(), memoryUsageTimeline.end());
    
    for (size_t i = 0; i < memoryUsageTimeline.size(); i++) {
        std::string nodeName = (i < liveness->topoSortedNodes.size()) ? 
                               liveness->topoSortedNodes[i]->id : "END";
        
        // Scale width to represent memory usage
        int width = (memoryUsageTimeline[i] * poolWidth) / (maxMemory > 0 ? maxMemory : 1);
        
        outFile << "    <div style='display: flex; align-items: center; margin-bottom: 5px;'>\n"
                << "      <div style='width: 150px; overflow: hidden; text-overflow: ellipsis;'>" 
                << nodeName << "</div>\n"
                << "      <div class='timeline-bar' style='width: " << width << "px;'></div>\n"
                << "      <div style='margin-left: 10px;'>" 
                << (memoryUsageTimeline[i] / 1024.0) << " KB</div>\n"
                << "    </div>\n";
    }
    
    outFile << "  </div>\n"
            << "</body>\n"
            << "</html>\n";
    
    outFile.close();
    
    llvm::outs() << "Generated memory usage visualization to " << filename << "\n";
}

size_t MemoryPlanner::getTotalMemoryUsage() const {
    return totalMemory;
}

size_t MemoryPlanner::getPeakMemoryUsage() const {
    return peakMemory;
}