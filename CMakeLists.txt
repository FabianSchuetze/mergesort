cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json

project(GPU_Mergesort LANGUAGES CUDA CXX)
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})

set(CUDA_HOST_COMPILER /usr/local/cuda-10.2/bin/nvcc)
set(CMAKE_CUDA_FLAGS "-Xcompiler -rdynamic -lineinfo")
enable_language(CUDA)
ADD_LIBRARY(cuda_obj SHARED src/mergesort.cu
                            src/copy.cu)

set(${LIBRARY_TARGET_NAME}_SRC
    src/merge.cpp
    src/storage.cpp
)

if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fPIC\
    -Wall -std=c++17 -Wno-error=unused-command-line-argument\
    -lcurand -L /usr/lib/cuda/lib64 -lcudart  -lcublas")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-g -03")

add_subdirectory(examples)
