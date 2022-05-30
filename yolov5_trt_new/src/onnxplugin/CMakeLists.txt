# 指定CMake版本
cmake_minimum_required(VERSION 3.5)
# 指定项目名称
project(onnxplugin)

# 指定头文件目录
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# 指定源文件目录
# GLOB_RECURSE参数：捕获所有的cpp文件作为SOURCE_FILES

file(GLOB_RECURSE cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE cuda_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# 打印cmake当前目录地址&源文件目录地址
message(module_CMAKE_CURRENT_SOURCE_DIR => ${CMAKE_CURRENT_SOURCE_DIR})
message( module_CPP_FILES => ${cpp_srcs})
message( module_CUDA_FILES => ${cuda_srcs})
 
# 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，
# 但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
set(ALL_SRCS ${cpp_srcs} ${cuda_srcs})
message(module_ALL_SRCS => ${ALL_SRCS})
 
#设置生成库保存位置
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 生成so包
# SHARED ->生成动态库
# STATIC ->生成静态库
message(module_PROJECT_NAME => ${PROJECT_NAME})
cuda_add_library(${PROJECT_NAME} SHARED ${ALL_SRCS})
# 链接相关cuda库
# find_library(CUDNN_LIB cudnn HINTS
#     ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
# find_library(CUBLAS_LIB cublas HINTS
#     ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
# find_library(CUBLASLT_LIB cublasLt HINTS
#     ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
# find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
# target_link_libraries(${PROJECT_NAME}
#     ${CUBLAS_LIB}
#     ${CUBLASLT_LIB}
#     ${CUDART_LIB}
#     ${CUDNN_LIB}
#     ${CMAKE_DL_LIBS}
# )
