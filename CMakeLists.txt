cmake_minimum_required(VERSION 3.5)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json

project(GPU_Mergesort LANGUAGES CUDA CXX)
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})

set(CUDA_HOST_COMPILER nvcc)
enable_language(CUDA)
ADD_LIBRARY(cuda_obj SHARED src/merge.cu
                            src/my_mergesort.cu) # works for me now

set(${LIBRARY_TARGET_NAME}_SRC
    src/merge.cpp
    src/sedgewick.cpp
    src/mergesort.cpp
    src/storage.cpp
)

if (UNIX)
  add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
endif(UNIX)

# moved the CXX flags to after the cuda_add_library call
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC\
    -Wall -std=c++17 -Wno-error=unused-command-line-argument\
    -lcurand -L /usr/lib/cuda/lib64 -lcudart  -lcublas")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

add_subdirectory(examples)
