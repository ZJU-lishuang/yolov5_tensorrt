#检查cmake版本
cmake_minimum_required(VERSION 3.5)  
#项目名
project(yolov5tensorrt)
#可执行文件保存目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_GEN_CODE "-gencode=arch=compute_86,code=sm_86")

#cuda
find_package(CUDA REQUIRED)
#自定义opencv路径
set(OpenCV_DIR /home/ls/softwares/opencv-4.5.5/build)
# find opencv
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(ERROR "OpenCV not found!")
endif(NOT OpenCV_FOUND)
#tensorrt
# set(TensorRT_DIR /home/ls/softwares/TensorRT-8.2.3.0)
set(TensorRT_DIR /home/ls/softwares/TensorRT)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_DIR}/include
)

link_directories(
    ${TensorRT_DIR}/lib
    /usr/local/cuda/lib64
)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

# 链接子项目部件
add_subdirectory(${CMAKE_SOURCE_DIR}/src/module)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/application)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/onnxplugin)
# include_directories(${CMAKE_SOURCE_DIR}/src/onnxplugin/include)
# 链接库目录
link_directories(${CMAKE_SOURCE_DIR}/lib)
# 链接依赖库
link_libraries(module)
link_libraries(application)
link_libraries(onnxplugin)

ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser)
target_link_libraries(${PROJECT_NAME} cuda cublas cudart cudnn)
target_link_libraries(${PROJECT_NAME} pthread)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})








