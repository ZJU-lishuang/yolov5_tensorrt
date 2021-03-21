#检查cmake版本
cmake_minimum_required(VERSION 3.5)  
#项目名
project(yolov5tensorrt)

#自定义opencv路径
# set(OpenCV_DIR /home/ls/softwares/opencv-4.5.1/build)
# find opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

if(NOT OpenCV_FOUND)
    message(ERROR "OpenCV not found!")
endif(NOT OpenCV_FOUND)
#cuda
find_package(CUDA REQUIRED)
# enable_language(CUDA)
message(STATUS "CUDA_LIBRARIES => ${CUDA_LIBRARIES}")
message(STATUS "CUDA_INCLUDE_DIRS => ${CUDA_INCLUDE_DIRS}")
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64" "/usr/lib" /usr/lib/x86_64-linux-gnu )
#tensorrt
set(TensorRT_DIR /home/ls/softwares/TensorRT-7.2.3.4)
if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "TensorRT Header => ${tensorrt_dir}/include")
message(STATUS "TensorRT Lib => ${tensorrt_dir}/lib")

#本地头文件
set(project_dir /home/ls/projects/yolov5_tensorrt)
#链接头文件目录
include_directories(${project_dir}/include)
#链接库目录
link_directories(${project_dir}/lib)
#设置生成库保存位置
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${project_dir}/lib)

## Output
#set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
## Project
#include_directories(${PROJECT_SOURCE_DIR}/include)
#link_directories(${PROJECT_SOURCE_DIR}/src)
#link_directories(${PROJECT_SOURCE_DIR}/lib)

# Loggers
set(tensorrtlib  src/logger.cpp)
set(YOLO5 ${tensorrtlib} src/yolov5.cpp)

set(TRT_LIB libnvinfer.so libnvonnxparser.so cudart.so)

#生成动态库文件so
cuda_add_library(yolo5trt SHARED ${YOLO5})

ADD_EXECUTABLE(yolov5tensorrt main.cpp)
target_link_libraries( yolov5tensorrt yolo5trt.so ${TRT_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES})








