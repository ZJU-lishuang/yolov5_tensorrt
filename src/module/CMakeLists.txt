# 指定CMake版本
cmake_minimum_required(VERSION 3.5)
# 指定项目名称
project(module)

# 指定头文件目录
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# 指定源文件目录
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# 打印cmake当前目录地址&源文件目录地址
message(module_CMAKE_CURRENT_SOURCE_DIR => ${CMAKE_CURRENT_SOURCE_DIR})
# message( module_SOURCE_FILES => ${SOURCE_FILES})
 
# 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，
# 但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
set(ALL_SRCS ${SOURCE_FILES})
# message(module_ALL_SRCS => ${ALL_SRCS})
 
#设置生成库保存位置
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

# 生成so包
# SHARED ->生成动态库
# STATIC ->生成静态库
message(module_PROJECT_NAME => ${PROJECT_NAME})
add_library(${PROJECT_NAME} SHARED ${ALL_SRCS})