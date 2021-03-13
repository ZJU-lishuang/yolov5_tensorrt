#检查cmake版本
cmake_minimum_required(VERSION 3.5)  
#项目名
project(yolov5tensorrt)

# output 设置生成可执行文件保存位置
# set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
# message(STATUS "Project_binary_dir : ${PROJECT_BINARY_DIR}")

#自定义opencv路径
set(OpenCV_DIR /home/lishuang/Disk/remote/opencv-4.4.0/build)
# find opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV_INCLUDE_DIRS => ${OpenCV_INCLUDE_DIRS}")

if(NOT OpenCV_FOUND)
    message(ERROR "OpenCV not found!")
endif(NOT OpenCV_FOUND)
#cuda
find_package(CUDA REQUIRED)
enable_language(CUDA)
message(STATUS "CUDA_LIBRARIES => ${CUDA_LIBRARIES}")
message(STATUS "CUDA_INCLUDE_DIRS => ${CUDA_INCLUDE_DIRS}")
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64" "/usr/lib" /usr/lib/x86_64-linux-gnu )
#tensorrt
set(TensorRT_DIR /home/lishuang/Disk/software/TensorRT-7.0.0.11)
if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "TensorRT Header => ${tensorrt_dir}/include")
message(STATUS "TensorRT Lib => ${tensorrt_dir}/lib")

#本地头文件
set(project_dir /home/lishuang/Disk/remote/yolov5-tensorrt)
#链接头文件目录
include_directories(${project_dir}/include)
#链接库目录
link_directories(${project_dir}/lib)


# AUX_SOURCE_DIRECTORY(. SRC_LIST)
#设置生成库保存位置
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${project_dir}/lib)
# Loggers
set(tensorrtlib  src/logger.cpp)

set(TRT_LIB libnvinfer.so libnvonnxparser.so cudart.so)

#生成动态库文件so
cuda_add_library(yolo5trt SHARED ${tensorrtlib})

# ADD_EXECUTABLE(yolov5tensorrt ${SRC_LIST})
ADD_EXECUTABLE(yolov5tensorrt main.cpp)

target_link_libraries( yolov5tensorrt yolo5trt.so ${TRT_LIB} ${OpenCV_LIBS} ${CUDA_LIBRARIES} )


# set_property(GLOBAL PROPERTY USE_FOLDERS on)

# # output
# set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")
# message(STATUS "Project_binary_dir : ${PROJECT_BINARY_DIR}")
# # c++ 11
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# # tensorRT
# #set(tensorrt_dir /work/TensorRT-6.0.1.5)
# set(tensorrt_dir /home/lishuang/Disk/software/TensorRT-7.0.0.11)
# set(project_dir /home/lishuang/Disk/remote/tensorRT-master)
# include_directories(${tensorrt_dir}/include)
# include_directories(${project_dir}/include)
# link_directories(${tensorrt_dir}/lib)
# link_directories(${project_dir}/source)
# link_directories(${project_dir}/lib)

# # Loggers
# aux_source_directory(${common_dir}/source common_src)
# set(COMMON_SRC ${common_src} CACHE INTERNAL "common_source" )
# set(LOGGER_SRC ${common_dir}/source/logger.cpp CACHE INTERNAL "logger" )

# message(STATUS "TensorRT Header => ${tensorrt_dir}/include")
# message(STATUS "TensorRT Lib => ${tensorrt_dir}/lib")

# # find opencv
# set(OpenCV_DIR /home/lishuang/Disk/remote/opencv-4.4.0/build)
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# if(NOT OpenCV_LIBRARY_DIRS)
#     set(OpenCV_LIBRARY_DIRS /usr/local/lib)
#     message(WARING " Can not find opencv lib. It will use the default path => ${OpenCV_LIBRARY_DIRS}")
# endif()
# link_directories(${OpenCV_LIBRARY_DIRS})
# message(STATUS "OpenCV_INCLUDE_DIRS => ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV_LIBRARY_DIRS => ${OpenCV_LIBRARY_DIRS}")

# if(NOT OpenCV_FOUND)
#     message(ERROR "OpenCV not found!")
# endif(NOT OpenCV_FOUND)

# # find cuda
# find_package(CUDA)
# find_package(CUDA REQUIRED)

# #include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(/usr/local/cuda/targets/x86_64-linux/include)
# if(NOT CUDA_LIBRARY_DIRS)
#     set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)
#     message(WARING " Can not find CUDA lib. It will use the default path => ${CUDA_LIBRARY_DIRS}")
# endif()
# link_directories(${CUDA_LIBRARY_DIRS})
# message(STATUS "CUDA_INCLUDE_DIRS : ${CUDA_INCLUDE_DIRS}")
# message(STATUS "CUDA_LIBRARY_DIRS : ${CUDA_LIBRARY_DIRS}")

# ###############################################
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${project_dir}/lib)
# set(TRT source/tensorrt.cpp source/logger.cpp source/utils.cpp source/utils.cu)
# set(INT8 source/Int8Calibrator.cu source/Int8Calibrator.cpp)

# set(YOLO5 ${TRT} ${INT8} source/yolov5.cpp)

# set(CV_LIB libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so)
# set(TRT_LIB libnvinfer.so libnvonnxparser.so cudart.so)

# ###############################################
# cuda_add_library(yolo5trt SHARED ${YOLO5})

# cuda_add_executable(yolo5 yolov5_main.cpp)
# target_link_libraries(yolo5 yolo5trt.so ${TRT_LIB} ${CV_LIB})







