#ifndef common_h
#define common_h
#include <eigen-git-mirror/Eigen/Core>
#include <fstream>
#include <iomanip>
#include <utility>
//#include <iostream>

#define MY_CHECK(call)                                             \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess) {                                \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
        }                                                          \
    }

#define CHECK_CUBLAS(call)                                                   \
    {                                                                        \
        cublasStatus_t err;                                                  \
        if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                       \
            fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, \
                    __LINE__);                                               \
            exit(1);                                                         \
        }                                                                    \
    }

#define CHECK_CURAND(call)                                                   \
    {                                                                        \
        curandStatus_t err;                                                  \
        if ((err = (call)) != CURAND_STATUS_SUCCESS) {                       \
            fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, \
                    __LINE__);                                               \
            exit(1);                                                         \
        }                                                                    \
    }
#endif
