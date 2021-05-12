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
set(TensorRT_DIR /home/ls/softwares/TensorRT)
if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "TensorRT Header => ${tensorrt_dir}/include")
message(STATUS "TensorRT Lib => ${tensorrt_dir}/lib")

set(TRT_LIB libnvinfer.so libnvonnxparser.so cudart.so)

add_subdirectory(${CMAKE_SOURCE_DIR}/module)
include_directories(${CMAKE_SOURCE_DIR}/module/include)
# 链接库目录
link_directories(${CMAKE_SOURCE_DIR}/lib)
# 链接依赖库
link_libraries(module)

#plugin
include_directories(${CMAKE_SOURCE_DIR}/plugin)
#设置生成库保存位置
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(SiLU ${PROJECT_SOURCE_DIR}/plugin/SiLU.cu ${PROJECT_SOURCE_DIR}/plugin/SiLUPlugin.cpp)
cuda_add_library(myplugins SHARED ${SiLU})
find_library(CUDNN_LIB cudnn HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
target_link_libraries(myplugins
    ${CUBLAS_LIB}
    ${CUBLASLT_LIB}
    ${CUDART_LIB}
    ${CUDNN_LIB}
    libnvinfer.so
    libnvonnxparser.so
    ${CMAKE_DL_LIBS}
)

ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} myplugins ${TRT_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES})









