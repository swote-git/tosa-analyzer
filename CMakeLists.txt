cmake_minimum_required(VERSION 3.13.4)
project(TosaAnalyzer)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER clang++)

# 컴파일 데이터베이스 생성 (IntelliSense 향상)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLVM 및 MLIR 경로 설정 (홈 디렉토리에 있다고 가정)
set(LLVM_DIR $ENV{HOME}/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR $ENV{HOME}/llvm-project/build/lib/cmake/mlir)

# LLVM 패키지 찾기
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# MLIR 패키지 찾기
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR in: ${MLIR_DIR}")

# LLVM/MLIR 헤더 포함 경로 추가
include_directories(
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# LLVM/MLIR 매크로 및 정의 추가
add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

# 소스 파일 추가
file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

# 필요한 LLVM 컴포넌트만 명시적으로 선택
llvm_map_components_to_libnames(llvm_libs support core irreader)

# 필요한 MLIR 라이브러리만 명시적으로 선택
set(MLIR_LIBS
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    ${CMAKE_HOME_DIRECTORY}/../llvm-project/build/lib/libMLIRTosaDialect.a 
    MLIRFuncDialect
)

# 라이브러리 링크 (명시적 방식)
target_link_libraries(${PROJECT_NAME}
    ${llvm_libs}
    ${MLIR_LIBS}
)