cmake_minimum_required(VERSION 3.13.4)
project(TOSAAnalyzer)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER clang++)

# 컴파일 데이터베이스 생성 (IntelliSense 향상)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLVM 및 MLIR 경로 설정 (홈 디렉토리에 있다고 가정)
set(LLVM_DIR $ENV{HOME}/llvm-project/build/lib/cmake/llvm)
set(MLIR_DIR $ENV{HOME}/llvm-project/build/lib/cmake/mlir)

# 명시적으로 라이브러리 경로 추가
link_directories(
    $ENV{HOME}/llvm-project/build/lib
)

# LLVM 패키지 찾기
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# MLIR 패키지 찾기
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR in: ${MLIR_DIR}")

# LLVM 컴포넌트 추가
llvm_map_components_to_libnames(llvm_libs support core irreader)

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

# MLIR 라이브러리 목록 가져오기
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)

# 라이브러리 링크 (자동 방식)
target_link_libraries(${PROJECT_NAME}
    ${llvm_libs}
    ${mlir_libs}
)

# 대안: 수동으로 라이브러리 지정 (위의 자동 방식이 실패할 경우 이 부분을 사용)
# target_link_libraries(${PROJECT_NAME}
#    ${llvm_libs}
#    MLIRAnalysis
#    MLIRDialect
#    MLIRIR
#    MLIRParser
#    MLIRPass
#    MLIRSupport
#    MLIRTransforms
# )